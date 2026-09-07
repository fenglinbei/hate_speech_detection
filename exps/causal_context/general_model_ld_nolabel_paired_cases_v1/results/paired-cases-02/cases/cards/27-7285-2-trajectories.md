# 查询 7285：预测轨迹

主桶：H_joint_only；focus task：hate。

全部候选标签：["H_joint_only:hate","Stable_wrong:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："hate"；四位轨迹：0001；六位轨迹：000101。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | false | 12.816211700439453 | -12.816211700439453 | 1 | false | 114 | 0 |
| CLnew | ["non-hate"] | false | 15.450756072998047 | -15.450756072998047 | 1 | false | 198 | 84 |
| CD | ["non-hate"] | false | 0.4474639892578125 | -0.4474639892578125 | 1 | false | 486 | 0 |
| CLDnew | ["hate"] | true | 1.4780807495117188 | 1.4780807495117188 | 1 | false | 570 | 84 |
| CLnewNoCat | ["non-hate"] | false | 15.0391845703125 | -15.0391845703125 | 1 | false | 186 | 72 |
| CLDnewNoCat | ["hate"] | true | 2.378662109375 | 2.378662109375 | 1 | false | 558 | 72 |

配对连续读数：{"E_S_given_D":2.8261260986328125,"E_S_given_D_hate_logodds":2.8261260986328125,"E_remove_with_D":0.9005813598632812,"E_remove_with_D_hate_logodds":0.9005813598632812,"E_remove_without_D":0.4115715026855469,"E_remove_without_D_hate_logodds":0.4115715026855469,"I_S_D":5.049098968505859,"I_S_D_hate_logodds":5.049098968505859}

## group

Gold：["Racism"]；四位轨迹：0000；六位轨迹：000000。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["Sexism"] | false | 5.560981750488281 | -38.72395324707031 | 1 | false | 187 | 0 |
| CLnew | [] | false | 10.11550521850586 | -51.28154373168945 | 1 | false | 271 | 84 |
| CD | ["Sexism"] | false | 5.358814239501953 | -26.930217742919922 | 1 | false | 568 | 0 |
| CLDnew | ["Sexism"] | false | 16.613412857055664 | -28.249042510986328 | 1 | false | 652 | 84 |
| CLnewNoCat | [] | false | 20.022533416748047 | -51.12969207763672 | 1 | false | 259 | 72 |
| CLDnewNoCat | ["Sexism"] | false | 15.070066452026367 | -27.17254638671875 | 1 | false | 640 | 72 |

配对连续读数：{"E_S_given_D":-0.24232864379882812,"E_remove_with_D":1.0764961242675781,"E_remove_without_D":0.15185165405273438,"I_S_D":12.163410186767578}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
