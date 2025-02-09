import mlx.nn as nn
import mlx.core as mx



class ParallelLayer(nn.Module):
    def __init__(self, world_size: int, rank: int, row_size: int, col_size: int) -> None:
        super().__init__()
        self.world_size = world_size
        self.rank = rank
        self.layer = nn.Linear(row_size, col_size // self.world_size, bias=False)

    def load_weights(self, file_or_weights, strict = True):
        new_file_or_weights = []
        for k, v in file_or_weights:
            new_file_or_weights.append((k, mx.split(v, self.world_size, axis=0)[self.rank]))
        return super().load_weights(new_file_or_weights, strict)

    def __call__(self, x: int):
        return self.layer(x)
    
class BaseLayer(nn.Module):
    def __init__(self, row_size: int, col_size: int) -> None:
        super().__init__()
        self.layer = nn.Linear(row_size, col_size, bias=False)

    def __call__(self, x: int):
        return self.layer(x)
    
if __name__ == '__main__':
    world = mx.distributed.init()
    # x = mx.distributed.all_sum(mx.ones(10))
    print("world rank", world.rank())
    weights = mx.load("layer.npz")
    print("weights", weights.keys())

    if world.rank() == 0:
        layer = BaseLayer(10, 20)
        layer.load_weights(list(weights.items()))
        x = layer(mx.ones(10))
        print("BaseLayer x", x, x.shape)

    layer = ParallelLayer(world.size(), world.rank(), 10, 20)
    layer.load_weights(list(weights.items()))
    x = layer(mx.ones(10))

    parallel_x = mx.distributed.all_gather(x)
    # parallel_x = mx.distributed.all_sum(x)
    if world.rank() == 0:
        print("ParallelLayer x", x, x.shape)
    print("ParallelLayer x", parallel_x, parallel_x.shape)

    # print("ParallelLayer x", x, x.shape)
    
