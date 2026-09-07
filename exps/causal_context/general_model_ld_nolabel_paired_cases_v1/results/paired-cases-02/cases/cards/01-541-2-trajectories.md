# 查询 541：预测轨迹

主桶：H_rescue；focus task：hate。

全部候选标签：["H_rescue:hate","Stable_wrong:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："non-hate"；四位轨迹：1111；六位轨迹：111011。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | true | 6.627841949462891 | 6.627841949462891 | 1 | false | 126 | 0 |
| CLnew | ["non-hate"] | true | 6.243564605712891 | 6.243564605712891 | 1 | false | 192 | 66 |
| CD | ["non-hate"] | true | 0.6117820739746094 | 0.6117820739746094 | 1 | false | 864 | 0 |
| CLDnew | ["hate"] | false | 0.6081047058105469 | -0.6081047058105469 | 1 | false | 930 | 66 |
| CLnewNoCat | ["non-hate"] | true | 7.737434387207031 | 7.737434387207031 | 1 | false | 180 | 54 |
| CLDnewNoCat | ["non-hate"] | true | 0.579376220703125 | 0.579376220703125 | 1 | false | 918 | 54 |

配对连续读数：{"E_S_given_D":-0.032405853271484375,"E_S_given_D_hate_logodds":0.032405853271484375,"E_remove_with_D":1.1874809265136719,"E_remove_with_D_hate_logodds":-1.1874809265136719,"E_remove_without_D":1.4938697814941406,"E_remove_without_D_hate_logodds":-1.4938697814941406,"I_S_D":-1.141998291015625,"I_S_D_hate_logodds":1.141998291015625}

## group

Gold：[]；四位轨迹：0000；六位轨迹：000000。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["LGBTQ"] | false | 9.057682037353516 | -11.102127075195312 | 1 | false | 199 | 0 |
| CLnew | ["LGBTQ"] | false | 9.942258834838867 | -9.942258834838867 | 1 | false | 265 | 66 |
| CD | ["LGBTQ","Sexism"] | false | 3.185710906982422 | -14.386634826660156 | 1 | false | 950 | 0 |
| CLDnew | ["LGBTQ","Sexism"] | false | 1.2838325500488281 | -13.943408966064453 | 1 | false | 1016 | 66 |
| CLnewNoCat | ["LGBTQ"] | false | 0.6721725463867188 | -0.6721725463867188 | 1 | true | 253 | 54 |
| CLDnewNoCat | ["LGBTQ","Sexism"] | false | 3.5985641479492188 | -15.403421401977539 | 1 | false | 1004 | 54 |

配对连续读数：{"E_S_given_D":-1.0167865753173828,"E_remove_with_D":-1.460012435913086,"E_remove_without_D":9.270086288452148,"I_S_D":-11.446741104125977}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
