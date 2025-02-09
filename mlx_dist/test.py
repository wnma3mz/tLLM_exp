import mlx.core as mx

world = mx.distributed.init()
x = mx.distributed.all_sum(mx.ones(10))
print(world.rank(), x)