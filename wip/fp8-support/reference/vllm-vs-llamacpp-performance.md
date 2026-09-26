---------

# VLLM PERFORMANCE

---------

## TP = 1

### Depth = 0

stew675@soar:~/cllm$ uvx llama-benchy --base-url http://localhost:8000/v1 --tg 128 --pp 512 --model Qwen3.5-4B-StewFP8 --tokenizer Qwen/Qwen3.5-4B
llama-benchy (0.4.0)
Date: 2026-08-05 17:42:41
Benchmarking model: Qwen3.5-4B-StewFP8 at http://localhost:8000/v1
Concurrency levels: [1]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading text from cache: /home/stew675/.cache/llama-benchy/cc6a0b5782734ee3b9069aa3b64cc62c.txt
Total tokens available in text corpus: 144480
Warming up...
Warmup (User only) complete. Delta: 9 tokens (Server: 30, Local: 21)
Warmup (System+Probe) complete. Delta: 14 tokens (Server: 36, Local context: 21, Probe: 1)

Running coherence test...
Coherence test PASSED.
Measuring latency using mode: api...
Average latency (api): 0.65 ms
Running test: pp=512, tg=128, depth=0, concurrency=1
  Warmup (batch size 1)...
  Run 1/3 (batch size 1)...
  Run 2/3 (batch size 1)...
  Run 3/3 (batch size 1)...
Printing results in MD format:


| model              |   test |               t/s |     peak t/s |    ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:-------------------|-------:|------------------:|-------------:|-------------:|---------------:|----------------:|
| Qwen3.5-4B-StewFP8 |  pp512 | 10455.82 ± 695.62 |              | 49.89 ± 3.17 |   49.24 ± 3.17 |    49.89 ± 3.17 |
| Qwen3.5-4B-StewFP8 |  tg128 |      84.50 ± 0.05 | 85.00 ± 0.00 |              |                |                 |

llama-benchy (0.4.0)
date: 2026-08-05 17:42:41 | latency mode: api


### Depth = 1024

stew675@soar:~/cllm$ uvx llama-benchy --base-url http://localhost:8000/v1 --tg 128 --pp 512 --model Qwen3.5-4B-StewFP8 --tokenizer Qwen/Qwen3.5-4B --depth 1024

| model              |          test |              t/s |     peak t/s |     ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:-------------------|--------------:|-----------------:|-------------:|--------------:|---------------:|----------------:|
| Qwen3.5-4B-StewFP8 | pp512 @ d1024 | 10814.42 ± 28.13 |              | 142.69 ± 0.37 |  142.13 ± 0.37 |   142.69 ± 0.37 |
| Qwen3.5-4B-StewFP8 | tg128 @ d1024 |     83.96 ± 0.04 | 84.67 ± 0.47 |               |                |                 |


### Depth = 2048

stew675@soar:~/cllm$ uvx llama-benchy --base-url http://localhost:8000/v1 --tg 128 --pp 512 --model Qwen3.5-4B-StewFP8 --tokenizer Qwen/Qwen3.5-4B --depth 2048

| model              |          test |              t/s |     peak t/s |     ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:-------------------|--------------:|-----------------:|-------------:|--------------:|---------------:|----------------:|
| Qwen3.5-4B-StewFP8 | pp512 @ d2048 | 10125.20 ± 63.83 |              | 253.48 ± 1.63 |  252.91 ± 1.63 |   253.48 ± 1.63 |
| Qwen3.5-4B-StewFP8 | tg128 @ d2048 |     83.34 ± 0.09 | 84.00 ± 0.00 |               |                |                 |


### Depth = 4096

stew675@soar:~/cllm$ uvx llama-benchy --base-url http://localhost:8000/v1 --tg 128 --pp 512 --model Qwen3.5-4B-StewFP8 --tokenizer Qwen/Qwen3.5-4B --depth 4096

| model              |          test |            t/s |     peak t/s |     ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:-------------------|--------------:|---------------:|-------------:|--------------:|---------------:|----------------:|
| Qwen3.5-4B-StewFP8 | pp512 @ d4096 | 9469.09 ± 6.48 |              | 487.45 ± 0.28 |  486.78 ± 0.28 |   487.45 ± 0.28 |
| Qwen3.5-4B-StewFP8 | tg128 @ d4096 |   82.50 ± 0.04 | 83.00 ± 0.00 |               |                |                 |


### Depth = 8192

stew675@soar:~/cllm$ uvx llama-benchy --base-url http://localhost:8000/v1 --tg 128 --pp 512 --model Qwen3.5-4B-StewFP8 --tokenizer Qwen/Qwen3.5-4B --depth 8192

| model              |          test |             t/s |     peak t/s |      ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:-------------------|--------------:|----------------:|-------------:|---------------:|---------------:|----------------:|
| Qwen3.5-4B-StewFP8 | pp512 @ d8192 | 7907.61 ± 16.91 |              | 1101.30 ± 2.41 | 1100.80 ± 2.41 |  1101.30 ± 2.41 |
| Qwen3.5-4B-StewFP8 | tg128 @ d8192 |    81.00 ± 0.12 | 81.67 ± 0.47 |                |                |                 |


### Depth = 16384

stew675@soar:~/cllm$ uvx llama-benchy --base-url http://localhost:8000/v1 --tg 128 --pp 512 --model Qwen3.5-4B-StewFP8 --tokenizer Qwen/Qwen3.5-4B --depth 16384

| model              |           test |             t/s |     peak t/s |       ttfr (ms) |    est_ppt (ms) |   e2e_ttft (ms) |
|:-------------------|---------------:|----------------:|-------------:|----------------:|----------------:|----------------:|
| Qwen3.5-4B-StewFP8 | pp512 @ d16384 | 6076.53 ± 27.63 |              | 2781.20 ± 12.69 | 2780.70 ± 12.69 | 2781.87 ± 12.74 |
| Qwen3.5-4B-StewFP8 | tg128 @ d16384 |    78.47 ± 0.06 | 79.00 ± 0.00 |                 |                 |                 |


### Depth = 32768

stew675@soar:~/cllm$ uvx llama-benchy --base-url http://localhost:8000/v1 --tg 128 --pp 512 --model Qwen3.5-4B-StewFP8 --tokenizer Qwen/Qwen3.5-4B --depth 32768

| model              |           test |            t/s |     peak t/s |      ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:-------------------|---------------:|---------------:|-------------:|---------------:|---------------:|----------------:|
| Qwen3.5-4B-StewFP8 | pp512 @ d32768 | 4075.66 ± 4.00 |              | 8166.39 ± 8.02 | 8165.80 ± 8.02 |  8167.58 ± 8.08 |
| Qwen3.5-4B-StewFP8 | tg128 @ d32768 |   73.81 ± 0.02 | 75.00 ± 0.00 |                |                |                 |


### Depth = 65536

stew675@soar:~/cllm$ uvx llama-benchy --base-url http://localhost:8000/v1 --tg 128 --pp 512 --model Qwen3.5-4B-StewFP8 --tokenizer Qwen/Qwen3.5-4B --depth 65536

| model              |           test |            t/s |     peak t/s |        ttfr (ms) |     est_ppt (ms) |    e2e_ttft (ms) |
|:-------------------|---------------:|---------------:|-------------:|-----------------:|-----------------:|-----------------:|
| Qwen3.5-4B-StewFP8 | pp512 @ d65536 | 2442.39 ± 1.10 |              | 27043.68 ± 12.03 | 27043.07 ± 12.03 | 27045.75 ± 12.02 |
| Qwen3.5-4B-StewFP8 | tg128 @ d65536 |   65.28 ± 0.03 | 67.00 ± 0.00 |                  |                  |                  |


### Depth = 131072

stew675@soar:~/cllm$ uvx llama-benchy --base-url http://localhost:8000/v1 --tg 128 --pp 512 --model Qwen3.5-4B-StewFP8 --tokenizer Qwen/Qwen3.5-4B --depth 131072

| model              |            test |            t/s |     peak t/s |        ttfr (ms) |     est_ppt (ms) |    e2e_ttft (ms) |
|:-------------------|----------------:|---------------:|-------------:|-----------------:|-----------------:|-----------------:|
| Qwen3.5-4B-StewFP8 | pp512 @ d131072 | 1342.70 ± 0.41 |              | 98000.64 ± 29.61 | 98000.14 ± 29.61 | 98004.80 ± 29.67 |
| Qwen3.5-4B-StewFP8 | tg128 @ d131072 |   53.59 ± 0.03 | 55.00 ± 0.00 |                  |                  |                  |

---------

## TP = 2

### Depth = 0

stew675@soar:~/cllm$ uvx llama-benchy --base-url http://localhost:8000/v1 --tg 128 --pp 512 --model Qwen3.5-4B-StewFP8 --tokenizer Qwen/Qwen3.5-4B

| model              |   test |              t/s |      peak t/s |    ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:-------------------|-------:|-----------------:|--------------:|-------------:|---------------:|----------------:|
| Qwen3.5-4B-StewFP8 |  pp512 | 10767.49 ± 50.15 |               | 48.27 ± 0.22 |   47.64 ± 0.22 |    48.27 ± 0.22 |
| Qwen3.5-4B-StewFP8 |  tg128 |    121.93 ± 0.27 | 122.67 ± 0.47 |              |                |                 |


### Depth = 1024

stew675@soar:~/cllm$ uvx llama-benchy --base-url http://localhost:8000/v1 --tg 128 --pp 512 --model Qwen3.5-4B-StewFP8 --tokenizer Qwen/Qwen3.5-4B --depth 1024

| model              |          test |              t/s |      peak t/s |     ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:-------------------|--------------:|-----------------:|--------------:|--------------:|---------------:|----------------:|
| Qwen3.5-4B-StewFP8 | pp512 @ d1024 | 12419.85 ± 52.65 |               | 124.39 ± 0.52 |  123.76 ± 0.52 |   124.39 ± 0.52 |
| Qwen3.5-4B-StewFP8 | tg128 @ d1024 |    121.30 ± 0.27 | 121.67 ± 0.47 |               |                |                 |


### Depth = 2048

stew675@soar:~/cllm$ uvx llama-benchy --base-url http://localhost:8000/v1 --tg 128 --pp 512 --model Qwen3.5-4B-StewFP8 --tokenizer Qwen/Qwen3.5-4B --depth 2048

| model              |          test |              t/s |      peak t/s |     ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:-------------------|--------------:|-----------------:|--------------:|--------------:|---------------:|----------------:|
| Qwen3.5-4B-StewFP8 | pp512 @ d2048 | 12278.10 ± 41.04 |               | 209.04 ± 0.73 |  208.56 ± 0.73 |   209.04 ± 0.73 |
| Qwen3.5-4B-StewFP8 | tg128 @ d2048 |    120.70 ± 0.34 | 121.00 ± 0.00 |               |                |                 |


### Depth = 4096

stew675@soar:~/cllm$ uvx llama-benchy --base-url http://localhost:8000/v1 --tg 128 --pp 512 --model Qwen3.5-4B-StewFP8 --tokenizer Qwen/Qwen3.5-4B --depth 4096

| model              |          test |             t/s |      peak t/s |     ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:-------------------|--------------:|----------------:|--------------:|--------------:|---------------:|----------------:|
| Qwen3.5-4B-StewFP8 | pp512 @ d4096 | 12129.61 ± 4.83 |               | 380.55 ± 0.15 |  379.98 ± 0.15 |   380.55 ± 0.15 |
| Qwen3.5-4B-StewFP8 | tg128 @ d4096 |   119.09 ± 0.22 | 119.67 ± 0.47 |               |                |                 |


### Depth = 8192

stew675@soar:~/cllm$ uvx llama-benchy --base-url http://localhost:8000/v1 --tg 128 --pp 512 --model Qwen3.5-4B-StewFP8 --tokenizer Qwen/Qwen3.5-4B --depth 8192

| model              |          test |              t/s |      peak t/s |     ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:-------------------|--------------:|-----------------:|--------------:|--------------:|---------------:|----------------:|
| Qwen3.5-4B-StewFP8 | pp512 @ d8192 | 10841.85 ± 45.86 |               | 803.51 ± 3.43 |  802.89 ± 3.43 |   803.51 ± 3.43 |
| Qwen3.5-4B-StewFP8 | tg128 @ d8192 |    116.68 ± 0.10 | 117.00 ± 0.00 |               |                |                 |


### Depth = 16384

stew675@soar:~/cllm$ uvx llama-benchy --base-url http://localhost:8000/v1 --tg 128 --pp 512 --model Qwen3.5-4B-StewFP8 --tokenizer Qwen/Qwen3.5-4B --depth 16384

| model              |           test |             t/s |      peak t/s |       ttfr (ms) |    est_ppt (ms) |   e2e_ttft (ms) |
|:-------------------|---------------:|----------------:|--------------:|----------------:|----------------:|----------------:|
| Qwen3.5-4B-StewFP8 | pp512 @ d16384 | 9006.46 ± 55.46 |               | 1876.76 ± 11.67 | 1876.17 ± 11.67 | 1877.44 ± 11.66 |
| Qwen3.5-4B-StewFP8 | tg128 @ d16384 |   113.27 ± 0.14 | 114.00 ± 0.00 |                 |                 |                 |


### Depth = 32768

stew675@soar:~/cllm$ uvx llama-benchy --base-url http://localhost:8000/v1 --tg 128 --pp 512 --model Qwen3.5-4B-StewFP8 --tokenizer Qwen/Qwen3.5-4B --depth 32768

| model              |           test |             t/s |      peak t/s |      ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:-------------------|---------------:|----------------:|--------------:|---------------:|---------------:|----------------:|
| Qwen3.5-4B-StewFP8 | pp512 @ d32768 | 6549.91 ± 10.07 |               | 5081.68 ± 7.82 | 5081.05 ± 7.82 |  5082.88 ± 7.82 |
| Qwen3.5-4B-StewFP8 | tg128 @ d32768 |   107.22 ± 0.03 | 108.00 ± 0.00 |                |                |                 |


### Depth = 65536

stew675@soar:~/cllm$ uvx llama-benchy --base-url http://localhost:8000/v1 --tg 128 --pp 512 --model Qwen3.5-4B-StewFP8 --tokenizer Qwen/Qwen3.5-4B --depth 65536

| model              |           test |            t/s |     peak t/s |        ttfr (ms) |     est_ppt (ms) |    e2e_ttft (ms) |
|:-------------------|---------------:|---------------:|-------------:|-----------------:|-----------------:|-----------------:|
| Qwen3.5-4B-StewFP8 | pp512 @ d65536 | 4246.70 ± 4.63 |              | 15553.64 ± 17.04 | 15552.95 ± 17.04 | 15555.90 ± 17.07 |
| Qwen3.5-4B-StewFP8 | tg128 @ d65536 |   95.69 ± 0.08 | 97.00 ± 0.00 |                  |                  |                  |


### Depth = 131072

stew675@soar:~/cllm$ uvx llama-benchy --base-url http://localhost:8000/v1 --tg 128 --pp 512 --model Qwen3.5-4B-StewFP8 --tokenizer Qwen/Qwen3.5-4B --depth 131072

| model              |            test |            t/s |     peak t/s |        ttfr (ms) |     est_ppt (ms) |    e2e_ttft (ms) |
|:-------------------|----------------:|---------------:|-------------:|-----------------:|-----------------:|-----------------:|
| Qwen3.5-4B-StewFP8 | pp512 @ d131072 | 2457.63 ± 2.21 |              | 53541.81 ± 48.50 | 53541.39 ± 48.50 | 53545.91 ± 48.59 |
| Qwen3.5-4B-StewFP8 | tg128 @ d131072 |   79.54 ± 0.06 | 82.00 ± 0.00 |                  |                  |                  |


---------

# LLAMA.CPP PERFORMANCE

---------

## TP=1

### Depth = 0

stew675@soar:~/llama.rocm$ uvx llama-benchy --base-url http://localhost:8080/v1 --tg 128 --pp 512 --model Qwen3.5-4B-Q8_0 --tokenizer Qwen/Qwen3.5-4B

| model           |   test |             t/s |     peak t/s |     ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:----------------|-------:|----------------:|-------------:|--------------:|---------------:|----------------:|
| Qwen3.5-4B-Q8_0 |  pp512 | 3629.13 ± 21.49 |              | 141.83 ± 0.74 |  141.45 ± 0.74 |   141.83 ± 0.74 |
| Qwen3.5-4B-Q8_0 |  tg128 |    89.23 ± 0.06 | 90.00 ± 0.00 |               |                |                 |


### Depth = 1024

stew675@soar:~/llama.rocm$ uvx llama-benchy --base-url http://localhost:8080/v1 --tg 128 --pp 512 --model Qwen3.5-4B-Q8_0 --tokenizer Qwen/Qwen3.5-4B --depth 1024

| model           |          test |             t/s |     peak t/s |     ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:----------------|--------------:|----------------:|-------------:|--------------:|---------------:|----------------:|
| Qwen3.5-4B-Q8_0 | pp512 @ d1024 | 4228.15 ± 16.76 |              | 364.05 ± 1.41 |  363.60 ± 1.41 |   364.05 ± 1.41 |
| Qwen3.5-4B-Q8_0 | tg128 @ d1024 |    88.52 ± 0.05 | 89.00 ± 0.00 |               |                |                 |


### Depth = 2048

| model           |          test |            t/s |     peak t/s |     ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:----------------|--------------:|---------------:|-------------:|--------------:|---------------:|----------------:|
| Qwen3.5-4B-Q8_0 | pp512 @ d2048 | 4602.07 ± 4.28 |              | 556.90 ± 0.52 |  556.49 ± 0.52 |   556.90 ± 0.52 |
| Qwen3.5-4B-Q8_0 | tg128 @ d2048 |   88.09 ± 0.05 | 89.00 ± 0.00 |               |                |                 |


### Depth = 4096

| model           |          test |            t/s |     peak t/s |     ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:----------------|--------------:|---------------:|-------------:|--------------:|---------------:|----------------:|
| Qwen3.5-4B-Q8_0 | pp512 @ d4096 | 4693.56 ± 9.45 |              | 982.42 ± 1.98 |  981.99 ± 1.98 |   982.42 ± 1.98 |
| Qwen3.5-4B-Q8_0 | tg128 @ d4096 |   86.97 ± 0.03 | 87.33 ± 0.47 |               |                |                 |


### Depth = 8192

| model           |          test |            t/s |     peak t/s |      ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:----------------|--------------:|---------------:|-------------:|---------------:|---------------:|----------------:|
| Qwen3.5-4B-Q8_0 | pp512 @ d8192 | 4473.35 ± 8.55 |              | 1946.40 ± 3.72 | 1946.05 ± 3.72 |  1946.40 ± 3.72 |
| Qwen3.5-4B-Q8_0 | tg128 @ d8192 |   85.50 ± 0.03 | 86.00 ± 0.00 |                |                |                 |


### Depth = 16384

| model           |           test |             t/s |     peak t/s |       ttfr (ms) |    est_ppt (ms) |   e2e_ttft (ms) |
|:----------------|---------------:|----------------:|-------------:|----------------:|----------------:|----------------:|
| Qwen3.5-4B-Q8_0 | pp512 @ d16384 | 3782.38 ± 27.73 |              | 4467.92 ± 32.66 | 4467.44 ± 32.66 | 4467.92 ± 32.66 |
| Qwen3.5-4B-Q8_0 | tg128 @ d16384 |    82.27 ± 0.04 | 83.00 ± 0.00 |                 |                 |                 |


### Depth = 32768

| model           |           test |            t/s |     peak t/s |        ttfr (ms) |     est_ppt (ms) |    e2e_ttft (ms) |
|:----------------|---------------:|---------------:|-------------:|-----------------:|-----------------:|-----------------:|
| Qwen3.5-4B-Q8_0 | pp512 @ d32768 | 2812.12 ± 3.56 |              | 11835.26 ± 15.15 | 11834.87 ± 15.15 | 11835.26 ± 15.15 |
| Qwen3.5-4B-Q8_0 | tg128 @ d32768 |   76.68 ± 0.03 | 77.00 ± 0.00 |                  |                  |                  |


### Depth = 65536

| model           |           test |            t/s |     peak t/s |       ttfr (ms) |    est_ppt (ms) |   e2e_ttft (ms) |
|:----------------|---------------:|---------------:|-------------:|----------------:|----------------:|----------------:|
| Qwen3.5-4B-Q8_0 | pp512 @ d65536 | 1879.03 ± 0.42 |              | 35151.25 ± 7.70 | 35150.83 ± 7.70 | 35151.25 ± 7.70 |
| Qwen3.5-4B-Q8_0 | tg128 @ d65536 |   67.83 ± 0.02 | 68.00 ± 0.00 |                 |                 |                 |



### Depth = 131072


| model           |            test |            t/s |     peak t/s |         ttfr (ms) |      est_ppt (ms) |     e2e_ttft (ms) |
|:----------------|----------------:|---------------:|-------------:|------------------:|------------------:|------------------:|
| Qwen3.5-4B-Q8_0 | pp512 @ d131072 | 1131.71 ± 0.20 |              | 116270.87 ± 20.89 | 116270.48 ± 20.89 | 116270.87 ± 20.89 |
| Qwen3.5-4B-Q8_0 | tg128 @ d131072 |   54.97 ± 0.00 | 55.00 ± 0.00 |                   |                   |                   |


---------

## TP=2

---------

### Depth = 0

$ uvx llama-benchy --base-url http://localhost:8000/v1 --tg 128 --pp 512 --model Qwen3.5-4B-Q8_0 --tokenizer Qwen/Qwen3.5-4B

| model           |   test |            t/s |     peak t/s |     ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:----------------|-------:|---------------:|-------------:|--------------:|---------------:|----------------:|
| Qwen3.5-4B-Q8_0 |  pp512 | 2930.88 ± 5.25 |              | 175.39 ± 0.31 |  175.03 ± 0.31 |   175.39 ± 0.31 |
| Qwen3.5-4B-Q8_0 |  tg128 |   95.67 ± 0.21 | 96.00 ± 0.00 |               |                |                 |


### Depth = 1024

$ uvx llama-benchy --base-url http://localhost:8000/v1 --tg 128 --pp 512 --model Qwen3.5-4B-Q8_0 --tokenizer Qwen/Qwen3.5-4B --depth 1024

| model           |          test |            t/s |     peak t/s |     ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:----------------|--------------:|---------------:|-------------:|--------------:|---------------:|----------------:|
| Qwen3.5-4B-Q8_0 | pp512 @ d1024 | 3701.95 ± 8.14 |              | 415.32 ± 0.96 |  414.92 ± 0.96 |   415.32 ± 0.96 |
| Qwen3.5-4B-Q8_0 | tg128 @ d1024 |   94.59 ± 0.53 | 95.00 ± 0.82 |               |                |                 |


### Depth = 2048

| model           |          test |             t/s |     peak t/s |     ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:----------------|--------------:|----------------:|-------------:|--------------:|---------------:|----------------:|
| Qwen3.5-4B-Q8_0 | pp512 @ d2048 | 4464.75 ± 24.72 |              | 573.94 ± 3.23 |  573.55 ± 3.23 |   573.94 ± 3.23 |
| Qwen3.5-4B-Q8_0 | tg128 @ d2048 |    94.41 ± 0.07 | 95.00 ± 0.00 |               |                |                 |


### Depth = 4096

| model           |          test |              t/s |     peak t/s |      ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:----------------|--------------:|-----------------:|-------------:|---------------:|---------------:|----------------:|
| Qwen3.5-4B-Q8_0 | pp512 @ d4096 | 4928.24 ± 185.01 |              | 936.89 ± 36.12 | 936.58 ± 36.12 |  936.89 ± 36.12 |
| Qwen3.5-4B-Q8_0 | tg128 @ d4096 |     92.95 ± 0.05 | 93.33 ± 0.47 |                |                |                 |


### Depth = 8192

| model           |          test |            t/s |     peak t/s |      ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:----------------|--------------:|---------------:|-------------:|---------------:|---------------:|----------------:|
| Qwen3.5-4B-Q8_0 | pp512 @ d8192 | 5268.36 ± 9.07 |              | 1652.86 ± 2.73 | 1652.51 ± 2.73 |  1652.86 ± 2.73 |
| Qwen3.5-4B-Q8_0 | tg128 @ d8192 |   92.13 ± 0.04 | 93.00 ± 0.00 |                |                |                 |


### Depth = 16384

| model           |           test |             t/s |     peak t/s |       ttfr (ms) |    est_ppt (ms) |   e2e_ttft (ms) |
|:----------------|---------------:|----------------:|-------------:|----------------:|----------------:|----------------:|
| Qwen3.5-4B-Q8_0 | pp512 @ d16384 | 4898.61 ± 30.01 |              | 3449.81 ± 21.13 | 3449.41 ± 21.13 | 3449.81 ± 21.13 |
| Qwen3.5-4B-Q8_0 | tg128 @ d16384 |    89.89 ± 0.17 | 90.33 ± 0.47 |                 |                 |                 |


### Depth = 32768

$ uvx llama-benchy --base-url http://localhost:8000/v1 --tg 128 --pp 512 --model Qwen3.5-4B-Q8_0 --tokenizer Qwen/Qwen3.5-4B --depth 32768

| model           |           test |             t/s |     peak t/s |       ttfr (ms) |    est_ppt (ms) |   e2e_ttft (ms) |
|:----------------|---------------:|----------------:|-------------:|----------------:|----------------:|----------------:|
| Qwen3.5-4B-Q8_0 | pp512 @ d32768 | 4124.81 ± 13.38 |              | 8068.81 ± 26.29 | 8068.50 ± 26.29 | 8068.81 ± 26.29 |
| Qwen3.5-4B-Q8_0 | tg128 @ d32768 |    86.50 ± 0.02 | 87.00 ± 0.00 |                 |                 |                 |


### Depth = 65536

| model           |           test |             t/s |     peak t/s |         ttfr (ms) |      est_ppt (ms) |     e2e_ttft (ms) |
|:----------------|---------------:|----------------:|-------------:|------------------:|------------------:|------------------:|
| Qwen3.5-4B-Q8_0 | pp512 @ d65536 | 2948.08 ± 13.34 |              | 22404.67 ± 101.39 | 22404.29 ± 101.39 | 22404.67 ± 101.39 |
| Qwen3.5-4B-Q8_0 | tg128 @ d65536 |    80.16 ± 0.70 | 80.67 ± 0.47 |                   |                   |                   |


### Depth = 131072

$ uvx llama-benchy --base-url http://localhost:8000/v1 --tg 128 --pp 512 --model Qwen3.5-4B-Q8_0 --tokenizer Qwen/Qwen3.5-4B --depth 131072

| model           |            test |            t/s |     peak t/s |        ttfr (ms) |     est_ppt (ms) |    e2e_ttft (ms) |
|:----------------|----------------:|---------------:|-------------:|-----------------:|-----------------:|-----------------:|
| Qwen3.5-4B-Q8_0 | pp512 @ d131072 | 1886.13 ± 1.33 |              | 69765.18 ± 48.95 | 69764.81 ± 48.95 | 69765.18 ± 48.95 |
| Qwen3.5-4B-Q8_0 | tg128 @ d131072 |   70.73 ± 0.02 | 71.00 ± 0.00 |                  |                  |                  |


---------
