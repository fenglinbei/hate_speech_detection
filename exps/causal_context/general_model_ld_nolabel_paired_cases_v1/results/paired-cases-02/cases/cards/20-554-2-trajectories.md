# 查询 554：预测轨迹

主桶：G_category_support；focus task：group。

全部候选标签：["G_category_support:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："non-hate"；四位轨迹：1100；六位轨迹：110010。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | true | 6.444591522216797 | 6.444591522216797 | 1 | false | 139 | 0 |
| CLnew | ["non-hate"] | true | 10.138851165771484 | 10.138851165771484 | 1 | false | 271 | 132 |
| CD | ["hate"] | false | 1.1558952331542969 | -1.1558952331542969 | 1 | false | 632 | 0 |
| CLDnew | ["hate"] | false | 0.4917144775390625 | -0.4917144775390625 | 1 | false | 764 | 132 |
| CLnewNoCat | ["non-hate"] | true | 8.311634063720703 | 8.311634063720703 | 1 | false | 247 | 108 |
| CLDnewNoCat | ["hate"] | false | 0.8381500244140625 | -0.8381500244140625 | 1 | false | 740 | 108 |

配对连续读数：{"E_S_given_D":0.3177452087402344,"E_S_given_D_hate_logodds":-0.3177452087402344,"E_remove_with_D":-0.346435546875,"E_remove_with_D_hate_logodds":0.346435546875,"E_remove_without_D":-1.8272171020507812,"E_remove_without_D_hate_logodds":1.8272171020507812,"I_S_D":-1.5492973327636719,"I_S_D_hate_logodds":1.5492973327636719}

## group

Gold：[]；四位轨迹：0000；六位轨迹：010000。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["Region"] | false | 3.0538177490234375 | -3.0538177490234375 | 1 | false | 212 | 0 |
| CLnew | [] | true | 1.6981658935546875 | 1.6981658935546875 | 1 | false | 344 | 132 |
| CD | ["Region"] | false | 12.765888214111328 | -16.655364990234375 | 1 | false | 718 | 0 |
| CLDnew | ["Region"] | false | 13.219856262207031 | -15.310291290283203 | 1 | false | 850 | 132 |
| CLnewNoCat | ["Region"] | false | 1.0925712585449219 | -1.0925712585449219 | 1 | false | 320 | 108 |
| CLDnewNoCat | ["Region"] | false | 14.20064926147461 | -14.20064926147461 | 1 | false | 826 | 108 |

配对连续读数：{"E_S_given_D":2.4547157287597656,"E_remove_with_D":1.1096420288085938,"E_remove_without_D":-2.7907371520996094,"I_S_D":0.49346923828125}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
