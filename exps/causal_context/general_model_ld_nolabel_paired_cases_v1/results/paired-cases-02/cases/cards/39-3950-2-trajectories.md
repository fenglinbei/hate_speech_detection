# 查询 3950：预测轨迹

主桶：Stable_correct；focus task：group。

全部候选标签：["H_residual:hate","Stable_correct:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："hate"；四位轨迹：0010；六位轨迹：001000。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | false | 2.1375694274902344 | -2.1375694274902344 | 1 | false | 148 | 0 |
| CLnew | ["non-hate"] | false | 2.7676544189453125 | -2.7676544189453125 | 1 | false | 567 | 419 |
| CD | ["hate"] | true | 1.6428604125976562 | 1.6428604125976562 | 1 | false | 870 | 0 |
| CLDnew | ["non-hate"] | false | 3.117443084716797 | -3.117443084716797 | 1 | false | 1289 | 419 |
| CLnewNoCat | ["non-hate"] | false | 2.5108871459960938 | -2.5108871459960938 | 1 | false | 515 | 367 |
| CLDnewNoCat | ["non-hate"] | false | 2.4315452575683594 | -2.4315452575683594 | 1 | false | 1237 | 367 |

配对连续读数：{"E_S_given_D":-4.074405670166016,"E_S_given_D_hate_logodds":-4.074405670166016,"E_remove_with_D":0.6858978271484375,"E_remove_with_D_hate_logodds":0.6858978271484375,"E_remove_without_D":0.25676727294921875,"E_remove_without_D_hate_logodds":0.25676727294921875,"I_S_D":-3.7010879516601562,"I_S_D_hate_logodds":-3.7010879516601562}

## group

Gold：["Sexism"]；四位轨迹：1111；六位轨迹：111111。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["Sexism"] | true | 11.333671569824219 | 11.333671569824219 | 1 | false | 221 | 0 |
| CLnew | ["Sexism"] | true | 9.985767364501953 | 9.985767364501953 | 1 | false | 640 | 419 |
| CD | ["Sexism"] | true | 17.5600528717041 | 17.5600528717041 | 1 | false | 953 | 0 |
| CLDnew | ["Sexism"] | true | 10.328399658203125 | 10.328399658203125 | 1 | false | 1372 | 419 |
| CLnewNoCat | ["Sexism"] | true | 6.042537689208984 | 6.042537689208984 | 1 | false | 588 | 367 |
| CLDnewNoCat | ["Sexism"] | true | 17.127838134765625 | 17.127838134765625 | 1 | false | 1320 | 367 |

配对连续读数：{"E_S_given_D":-0.43221473693847656,"E_remove_with_D":6.7994384765625,"E_remove_without_D":-3.9432296752929688,"I_S_D":4.858919143676758}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
