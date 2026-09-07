# 查询 3169：预测轨迹

主桶：H_residual；focus task：hate。

全部候选标签：["H_residual:hate"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："non-hate"；四位轨迹：1110；六位轨迹：111010。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | true | 15.286598205566406 | 15.286598205566406 | 1 | false | 115 | 0 |
| CLnew | ["non-hate"] | true | 1.5739173889160156 | 1.5739173889160156 | 1 | false | 399 | 284 |
| CD | ["non-hate"] | true | 4.859245300292969 | 4.859245300292969 | 1 | false | 495 | 0 |
| CLDnew | ["hate"] | false | 2.041271209716797 | -2.041271209716797 | 1 | false | 779 | 284 |
| CLnewNoCat | ["non-hate"] | true | 1.2375144958496094 | 1.2375144958496094 | 1 | false | 362 | 247 |
| CLDnewNoCat | ["hate"] | false | 3.7691802978515625 | -3.7691802978515625 | 1 | false | 742 | 247 |

配对连续读数：{"E_S_given_D":-8.628425598144531,"E_S_given_D_hate_logodds":8.628425598144531,"E_remove_with_D":-1.7279090881347656,"E_remove_with_D_hate_logodds":1.7279090881347656,"E_remove_without_D":-0.33640289306640625,"E_remove_without_D_hate_logodds":0.33640289306640625,"I_S_D":5.420658111572266,"I_S_D_hate_logodds":-5.420658111572266}

## group

Gold：[]；四位轨迹：1010；六位轨迹：101000。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | [] | true | 23.72215461730957 | 23.72215461730957 | 1 | false | 188 | 0 |
| CLnew | ["Racism"] | false | 11.558006286621094 | -11.558006286621094 | 1 | false | 472 | 284 |
| CD | [] | true | 5.255302429199219 | 5.255302429199219 | 1 | false | 573 | 0 |
| CLDnew | ["Racism"] | false | 18.698143005371094 | -18.698143005371094 | 1 | false | 857 | 284 |
| CLnewNoCat | ["Racism"] | false | 12.363639831542969 | -12.363639831542969 | 1 | false | 435 | 247 |
| CLDnewNoCat | ["Racism"] | false | 17.466989517211914 | -17.466989517211914 | 1 | false | 820 | 247 |

配对连续读数：{"E_S_given_D":-22.722291946411133,"E_remove_with_D":1.2311534881591797,"E_remove_without_D":-0.805633544921875,"I_S_D":13.363502502441406}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
