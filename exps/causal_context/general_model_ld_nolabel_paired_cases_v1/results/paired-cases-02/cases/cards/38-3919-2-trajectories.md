# 查询 3919：预测轨迹

主桶：Stable_correct；focus task：hate。

全部候选标签：["Stable_correct:hate"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："non-hate"；四位轨迹：1111；六位轨迹：111111。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | true | 7.633708953857422 | 7.633708953857422 | 1 | false | 148 | 0 |
| CLnew | ["non-hate"] | true | 6.621742248535156 | 6.621742248535156 | 1 | false | 465 | 317 |
| CD | ["non-hate"] | true | 2.843029022216797 | 2.843029022216797 | 1 | false | 875 | 0 |
| CLDnew | ["non-hate"] | true | 4.1696929931640625 | 4.1696929931640625 | 1 | false | 1192 | 317 |
| CLnewNoCat | ["non-hate"] | true | 6.367153167724609 | 6.367153167724609 | 1 | false | 432 | 284 |
| CLDnewNoCat | ["non-hate"] | true | 2.6846885681152344 | 2.6846885681152344 | 1 | false | 1159 | 284 |

配对连续读数：{"E_S_given_D":-0.1583404541015625,"E_S_given_D_hate_logodds":0.1583404541015625,"E_remove_with_D":-1.4850044250488281,"E_remove_with_D_hate_logodds":1.4850044250488281,"E_remove_without_D":-0.2545890808105469,"E_remove_without_D_hate_logodds":0.2545890808105469,"I_S_D":1.10821533203125,"I_S_D_hate_logodds":-1.10821533203125}

## group

Gold：[]；四位轨迹：1010；六位轨迹：101000。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | [] | true | 15.767807006835938 | 15.767807006835938 | 1 | false | 221 | 0 |
| CLnew | ["Region"] | false | 7.5254669189453125 | -7.5254669189453125 | 1 | false | 538 | 317 |
| CD | [] | true | 18.229440689086914 | 18.229440689086914 | 1 | false | 950 | 0 |
| CLDnew | ["Region"] | false | 15.342761993408203 | -19.19420623779297 | 1 | false | 1267 | 317 |
| CLnewNoCat | ["Region"] | false | 8.582752227783203 | -9.346450805664062 | 1 | false | 505 | 284 |
| CLDnewNoCat | ["Region"] | false | 20.835468292236328 | -20.835468292236328 | 1 | false | 1234 | 284 |

配对连续读数：{"E_S_given_D":-39.06490898132324,"E_remove_with_D":-1.6412620544433594,"E_remove_without_D":-1.82098388671875,"I_S_D":-13.950651168823242}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
