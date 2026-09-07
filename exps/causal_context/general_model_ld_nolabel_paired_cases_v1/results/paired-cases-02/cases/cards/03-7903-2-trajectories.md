# 查询 7903：预测轨迹

主桶：H_rescue；focus task：hate。

全部候选标签：["H_rescue:hate"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："non-hate"；四位轨迹：1111；六位轨迹：111011。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | true | 10.417671203613281 | 10.417671203613281 | 1 | false | 116 | 0 |
| CLnew | ["non-hate"] | true | 11.82358169555664 | 11.82358169555664 | 1 | false | 381 | 265 |
| CD | ["non-hate"] | true | 5.511539459228516 | 5.511539459228516 | 1 | false | 422 | 0 |
| CLDnew | ["hate"] | false | 0.1367950439453125 | -0.1367950439453125 | 1 | true | 687 | 265 |
| CLnewNoCat | ["non-hate"] | true | 16.549083709716797 | 16.549083709716797 | 1 | false | 345 | 229 |
| CLDnewNoCat | ["non-hate"] | true | 0.19975662231445312 | 0.19975662231445312 | 1 | false | 651 | 229 |

配对连续读数：{"E_S_given_D":-5.3117828369140625,"E_S_given_D_hate_logodds":5.3117828369140625,"E_remove_with_D":0.3365516662597656,"E_remove_with_D_hate_logodds":-0.3365516662597656,"E_remove_without_D":4.725502014160156,"E_remove_without_D_hate_logodds":-4.725502014160156,"I_S_D":-11.443195343017578,"I_S_D_hate_logodds":11.443195343017578}

## group

Gold：[]；四位轨迹：1100；六位轨迹：110010。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | [] | true | 11.278579711914062 | 11.278579711914062 | 1 | false | 189 | 0 |
| CLnew | [] | true | 15.805543899536133 | 15.805543899536133 | 1 | false | 454 | 265 |
| CD | ["Region"] | false | 11.747138977050781 | -11.747138977050781 | 1 | false | 493 | 0 |
| CLDnew | ["Region"] | false | 12.148571014404297 | -14.529287338256836 | 1 | false | 758 | 265 |
| CLnewNoCat | [] | true | 16.232568740844727 | 16.232568740844727 | 1 | false | 418 | 229 |
| CLDnewNoCat | ["Region"] | false | 5.911567687988281 | -15.763097763061523 | 1 | false | 722 | 229 |

配对连续读数：{"E_S_given_D":-4.015958786010742,"E_remove_with_D":-1.2338104248046875,"E_remove_without_D":0.42702484130859375,"I_S_D":-8.969947814941406}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
