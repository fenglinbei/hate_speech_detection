# 查询 7646：预测轨迹

主桶：H_removal_harm；focus task：hate。

全部候选标签：["H_removal_harm:hate"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："non-hate"；四位轨迹：1100；六位轨迹：110110。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | true | 11.27109146118164 | 11.27109146118164 | 1 | false | 120 | 0 |
| CLnew | ["non-hate"] | true | 12.822521209716797 | 12.822521209716797 | 1 | false | 310 | 190 |
| CD | ["hate"] | false | 1.725921630859375 | -1.725921630859375 | 1 | false | 384 | 0 |
| CLDnew | ["non-hate"] | true | 0.459320068359375 | 0.459320068359375 | 1 | false | 574 | 190 |
| CLnewNoCat | ["non-hate"] | true | 12.991649627685547 | 12.991649627685547 | 1 | false | 285 | 165 |
| CLDnewNoCat | ["hate"] | false | 1.7892913818359375 | -1.7892913818359375 | 1 | false | 549 | 165 |

配对连续读数：{"E_S_given_D":-0.0633697509765625,"E_S_given_D_hate_logodds":0.0633697509765625,"E_remove_with_D":-2.2486114501953125,"E_remove_with_D_hate_logodds":2.2486114501953125,"E_remove_without_D":0.16912841796875,"E_remove_without_D_hate_logodds":-0.16912841796875,"I_S_D":-1.7839279174804688,"I_S_D_hate_logodds":1.7839279174804688}

## group

Gold：[]；四位轨迹：1111；六位轨迹：111011。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | [] | true | 20.326702117919922 | 20.326702117919922 | 1 | false | 193 | 0 |
| CLnew | [] | true | 19.11904525756836 | 19.11904525756836 | 1 | false | 383 | 190 |
| CD | [] | true | 12.962997436523438 | 12.962997436523438 | 1 | false | 458 | 0 |
| CLDnew | ["others"] | false | 0.5384330749511719 | -1.5722770690917969 | 1 | false | 648 | 190 |
| CLnewNoCat | [] | true | 18.464622497558594 | 18.464622497558594 | 1 | false | 358 | 165 |
| CLDnewNoCat | [] | true | 4.291511535644531 | 4.291511535644531 | 1 | false | 623 | 165 |

配对连续读数：{"E_S_given_D":-8.671485900878906,"E_remove_with_D":5.863788604736328,"E_remove_without_D":-0.6544227600097656,"I_S_D":-6.809406280517578}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
