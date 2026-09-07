# 查询 7960：预测轨迹

主桶：G_category_support；focus task：group。

全部候选标签：["G_category_support:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："non-hate"；四位轨迹：1101；六位轨迹：110111。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | true | 4.181343078613281 | 4.181343078613281 | 1 | false | 144 | 0 |
| CLnew | ["non-hate"] | true | 5.135318756103516 | 5.135318756103516 | 1 | false | 459 | 315 |
| CD | ["hate"] | false | 1.911142349243164 | -1.911142349243164 | 1 | false | 879 | 0 |
| CLDnew | ["non-hate"] | true | 3.106456756591797 | 3.106456756591797 | 1 | false | 1194 | 315 |
| CLnewNoCat | ["non-hate"] | true | 2.669219970703125 | 2.669219970703125 | 1 | false | 419 | 275 |
| CLDnewNoCat | ["non-hate"] | true | 1.0322761535644531 | 1.0322761535644531 | 1 | false | 1154 | 275 |

配对连续读数：{"E_S_given_D":2.943418502807617,"E_S_given_D_hate_logodds":-2.943418502807617,"E_remove_with_D":-2.0741806030273438,"E_remove_with_D_hate_logodds":2.0741806030273438,"E_remove_without_D":-2.4660987854003906,"E_remove_without_D_hate_logodds":2.4660987854003906,"I_S_D":4.455541610717773,"I_S_D_hate_logodds":-4.455541610717773}

## group

Gold：[]；四位轨迹：1000；六位轨迹：110000。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | [] | true | 15.041690826416016 | 15.041690826416016 | 1 | false | 217 | 0 |
| CLnew | [] | true | 3.1986217498779297 | 3.1986217498779297 | 1 | true | 532 | 315 |
| CD | ["Sexism"] | false | 8.186588287353516 | -8.186588287353516 | 1 | false | 953 | 0 |
| CLDnew | ["others"] | false | 12.8541259765625 | -16.027114868164062 | 1 | false | 1268 | 315 |
| CLnewNoCat | ["Sexism"] | false | 7.9842987060546875 | -7.9842987060546875 | 1 | false | 492 | 275 |
| CLDnewNoCat | ["Sexism"] | false | 5.323497772216797 | -11.895187377929688 | 1 | false | 1228 | 275 |

配对连续读数：{"E_S_given_D":-3.708599090576172,"E_remove_with_D":4.131927490234375,"E_remove_without_D":-11.182920455932617,"I_S_D":19.31739044189453}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
