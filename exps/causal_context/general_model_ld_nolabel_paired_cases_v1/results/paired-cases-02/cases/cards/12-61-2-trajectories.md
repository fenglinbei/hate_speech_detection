# 查询 61：预测轨迹

主桶：H_removal_harm；focus task：hate。

全部候选标签：["H_removal_harm:hate","Stable_correct:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："hate"；四位轨迹：1010；六位轨迹：101100。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["hate"] | true | 0.6843795776367188 | 0.6843795776367188 | 1 | false | 113 | 0 |
| CLnew | ["non-hate"] | false | 6.867942810058594 | -6.867942810058594 | 1 | false | 428 | 315 |
| CD | ["hate"] | true | 4.850946426391602 | 4.850946426391602 | 1 | false | 807 | 0 |
| CLDnew | ["hate"] | true | 1.4790916442871094 | 1.4790916442871094 | 1 | false | 1122 | 315 |
| CLnewNoCat | ["non-hate"] | false | 8.853134155273438 | -8.853134155273438 | 1 | false | 391 | 278 |
| CLDnewNoCat | ["non-hate"] | false | 0.5401592254638672 | -0.5401592254638672 | 1 | false | 1085 | 278 |

配对连续读数：{"E_S_given_D":-5.391105651855469,"E_S_given_D_hate_logodds":-5.391105651855469,"E_remove_with_D":-2.0192508697509766,"E_remove_with_D_hate_logodds":-2.0192508697509766,"E_remove_without_D":-1.9851913452148438,"E_remove_without_D_hate_logodds":-1.9851913452148438,"I_S_D":4.1464080810546875,"I_S_D_hate_logodds":4.1464080810546875}

## group

Gold：["Racism"]；四位轨迹：1111；六位轨迹：111111。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["Racism"] | true | 16.653078079223633 | 16.653078079223633 | 1 | false | 186 | 0 |
| CLnew | ["Racism"] | true | 8.151885986328125 | 8.151885986328125 | 1 | false | 501 | 315 |
| CD | ["Racism"] | true | 23.23133087158203 | 23.23133087158203 | 1 | false | 890 | 0 |
| CLDnew | ["Racism"] | true | 22.107145309448242 | 22.107145309448242 | 1 | false | 1205 | 315 |
| CLnewNoCat | ["Racism"] | true | 3.313068389892578 | 3.313068389892578 | 1 | false | 464 | 278 |
| CLDnewNoCat | ["Racism"] | true | 17.807281494140625 | 17.807281494140625 | 1 | false | 1168 | 278 |

配对连续读数：{"E_S_given_D":-5.424049377441406,"E_remove_with_D":-4.299863815307617,"E_remove_without_D":-4.838817596435547,"I_S_D":7.915960311889648}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
