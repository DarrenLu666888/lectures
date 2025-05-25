import torch
from torch.profiler import profile, record_function, ProfilerActivity


# ## Default way to use profiler
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
#     for _ in range(10):
#         a = torch.square(torch.randn(10000, 10000).cuda())

# prof.export_chrome_trace("trace.json")


## With warmup and skip
# https://pytorch.org/docs/stable/profiler.html

# Non-default profiler schedule allows user to turn profiler on and off
# on different iterations of the training loop;
# trace_handler is called every time a new trace becomes available
def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace("tmp/test_trace_" + str(prof.step_num) + ".json")

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    # record_shapes=True,
    # profile_memory=True,
    # with_stack=True,
    # with_flops=True,
    # with_modules=True,

    # In this example with wait=1, warmup=1, active=2, repeat=1,
    # profiler will skip the first step/iteration,
    # start warming up on the second, record
    # the third and the forth iterations,
    # after which the trace will become available
    # and on_trace_ready (when set) is called;
    # the cycle repeats starting with the next step

    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2,
        repeat=1),
    on_trace_ready=trace_handler
    # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    # used when outputting for tensorboard
    ) as p:
        N=1000
        device = torch.device('cuda' if torch.cuda.is_available() else
                              'cpu')
        print(device)
        for iter in range(10):
            torch.matmul(torch.randn(N, N).to(device),torch.randn(N, N).to(device))
            # send a signal to the profiler that the next iteration has started
            torch.cuda.synchronize()  # 同步 CUDA 操作
            p.step()