import os
from typing import List

import torch
from fairscale.nn.pipe import microbatch
from torch import nn

import torch.distributed as torch_distrib
from torch.distributed import rpc

from pytorch_lightning import _logger as log
from fairscale.nn import Pipe
import fairscale.nn.model_parallel as mpu
from fairscale.nn.pipe.pipeline import PipelineStyle
from torch.nn.parallel import DistributedDataParallel

from pytorch_lightning import LightningModule
from pytorch_lightning.accelerators import DDPAccelerator
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin


def get_worker_map():
    return {rank: f"Test{rank}" for rank in range(torch_distrib.get_world_size())}


class LightningPipeModule(nn.Module):
    def __init__(self, module: nn.Sequential, layer_partitions: List[int], microbatches: int = 8):
        super().__init__()
        self.module = module
        self.layer_partitions = layer_partitions
        self.microbatches = microbatches

    def init_pipe(self):
        self.module = Pipe(
            module=self.module,
            balance=self.layer_partitions,
            chunks=self.microbatches,
            style=PipelineStyle.MultiProcess,
            worker_map=get_worker_map()
        )

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        if not self.training:
            x = microbatch.gather(x)
        return x


class PipeAccelerator(DDPAccelerator):
    def __init__(self, pipe_module: LightningPipeModule, trainer=None, cluster_environment=None, ddp_plugin=None):
        super().__init__(trainer, cluster_environment, ddp_plugin)
        self.pipe_module = pipe_module
        self.nickname = 'ddp_pipe'

    def init_ddp_connection(
            self, global_rank: int, world_size: int, is_slurm_managing_tasks: bool = True
    ) -> None:
        os.environ["MASTER_ADDR"] = str(self.cluster_environment.master_address())
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())
        os.environ["WORLD_SIZE"] = str(self.cluster_environment.world_size())

        torch_backend = "nccl" if self.trainer.on_gpu else "gloo"

        init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"

        if not torch.distributed.is_initialized():
            log.info(
                f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}"
            )
            torch_distrib.init_process_group(
                torch_backend, rank=global_rank, world_size=world_size, init_method=init_method
            )

        os.environ["MASTER_PORT"] = "10639"  # TODO change...
        init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"

        rpc.init_rpc(
            f"Test{global_rank}",
            rank=global_rank,
            world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(init_method=init_method),
        )
        mpu.initialize_model_parallel(model_parallel_size_=1, pipeline_length=len(self.pipe_module.layer_partitions))
        self.pipe_module.init_pipe()

    def configure_ddp(
            self, model: LightningModule, device_ids: List[int]
    ) -> DistributedDataParallel:
        self.ddp_plugin = DDPPlugin(process_group=mpu.get_data_parallel_group())
        model = self.ddp_plugin.configure_ddp(model, device_ids)
        return model

    def backward(self, closure_loss, optimizer, opt_idx, *args, **kwargs):
        if self.pipe_module.module.group.rank() == self.pipe_module.module.group.size() - 1:
            if self.trainer.precision == 16:
                closure_loss = self.trainer.precision_connector.backend.backward(
                    closure_loss, optimizer, opt_idx, *args, **kwargs
                )
            else:
                # do backward pass
                model = self.trainer.get_model()
                model.backward(closure_loss, optimizer, opt_idx, *args, **kwargs)

                # once backward has been applied, release graph
                closure_loss = closure_loss.detach()
            return closure_loss
        else:
            self.pipe_module.module.backward_helper(closure_loss)