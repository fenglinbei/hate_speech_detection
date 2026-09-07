# 查询 1128：预测轨迹

主桶：Stable_correct；focus task：hate。

全部候选标签：["Stable_correct:hate","Stable_correct:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："non-hate"；四位轨迹：1111；六位轨迹：111111。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | true | 15.603790283203125 | 15.603790283203125 | 1 | false | 109 | 0 |
| CLnew | ["non-hate"] | true | 18.814407348632812 | 18.814407348632812 | 1 | false | 186 | 77 |
| CD | ["non-hate"] | true | 11.54391860961914 | 11.54391860961914 | 1 | false | 414 | 0 |
| CLDnew | ["non-hate"] | true | 14.834182739257812 | 14.834182739257812 | 1 | false | 491 | 77 |
| CLnewNoCat | ["non-hate"] | true | 18.159069061279297 | 18.159069061279297 | 1 | false | 173 | 64 |
| CLDnewNoCat | ["non-hate"] | true | 14.820850372314453 | 14.820850372314453 | 1 | false | 478 | 64 |

配对连续读数：{"E_S_given_D":3.2769317626953125,"E_S_given_D_hate_logodds":-3.2769317626953125,"E_remove_with_D":-0.013332366943359375,"E_remove_with_D_hate_logodds":0.013332366943359375,"E_remove_without_D":-0.6553382873535156,"E_remove_without_D_hate_logodds":0.6553382873535156,"I_S_D":0.7216529846191406,"I_S_D_hate_logodds":-0.7216529846191406}

## group

Gold：[]；四位轨迹：1111；六位轨迹：111111。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | [] | true | 28.060550689697266 | 28.060550689697266 | 1 | false | 182 | 0 |
| CLnew | [] | true | 28.19965362548828 | 28.19965362548828 | 1 | false | 259 | 77 |
| CD | [] | true | 19.24746322631836 | 19.24746322631836 | 1 | false | 485 | 0 |
| CLDnew | [] | true | 19.456525802612305 | 19.456525802612305 | 1 | false | 562 | 77 |
| CLnewNoCat | [] | true | 29.909456253051758 | 29.909456253051758 | 1 | false | 246 | 64 |
| CLDnewNoCat | [] | true | 20.22385025024414 | 20.22385025024414 | 1 | false | 549 | 64 |

配对连续读数：{"E_S_given_D":0.9763870239257812,"E_remove_with_D":0.7673244476318359,"E_remove_without_D":1.7098026275634766,"I_S_D":-0.8725185394287109}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
