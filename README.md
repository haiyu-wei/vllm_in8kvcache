## INT8 KVCache

主要实现在：`vllm/v1/attention/backends/pagedint8.py - INT8PAttnImpl`

![image-20260307170024018](/imgs/1.png)



gather_kv_from_paged_cache 把paged attn的kvcache展开成连续tensor

![](/imgs/2.png)

量化函数（perhead）

![image-20260307161607821](/imgs/3.png)



pytorch算子：

![image-20260307165617272](//imgs/4.png)

![image-20260307165631274](/imgs/5.png)

![image-20260307165643017](/imgs/6.png)