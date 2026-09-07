# 查询 3898：预测轨迹

主桶：H_residual；focus task：hate。

全部候选标签：["H_residual:hate","Stable_correct:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："hate"；四位轨迹：0010；六位轨迹：001000。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | false | 1.9487228393554688 | -1.9487228393554688 | 1 | false | 171 | 0 |
| CLnew | ["non-hate"] | false | 0.6756706237792969 | -0.6756706237792969 | 1 | false | 579 | 408 |
| CD | ["hate"] | true | 2.3527889251708984 | 2.3527889251708984 | 1 | false | 622 | 0 |
| CLDnew | ["non-hate"] | false | 1.7487125396728516 | -1.7487125396728516 | 1 | false | 1030 | 408 |
| CLnewNoCat | ["non-hate"] | false | 0.3055267333984375 | -0.3055267333984375 | 1 | false | 506 | 335 |
| CLDnewNoCat | ["non-hate"] | false | 0.08431434631347656 | -0.08431434631347656 | 1 | false | 957 | 335 |

配对连续读数：{"E_S_given_D":-2.437103271484375,"E_S_given_D_hate_logodds":-2.437103271484375,"E_remove_with_D":1.664398193359375,"E_remove_with_D_hate_logodds":1.664398193359375,"E_remove_without_D":0.3701438903808594,"E_remove_without_D_hate_logodds":0.3701438903808594,"I_S_D":-4.080299377441406,"I_S_D_hate_logodds":-4.080299377441406}

## group

Gold：["Racism"]；四位轨迹：1111；六位轨迹：111111。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["Racism"] | true | 9.943166732788086 | 9.943166732788086 | 1 | false | 244 | 0 |
| CLnew | ["Racism"] | true | 5.384330749511719 | 5.384330749511719 | 1 | false | 652 | 408 |
| CD | ["Racism"] | true | 9.677837371826172 | 9.677837371826172 | 1 | false | 716 | 0 |
| CLDnew | ["Racism"] | true | 8.719207763671875 | 8.719207763671875 | 1 | false | 1124 | 408 |
| CLnewNoCat | ["Racism"] | true | 5.144374847412109 | 5.144374847412109 | 1 | false | 579 | 335 |
| CLDnewNoCat | ["Racism"] | true | 7.938877105712891 | 7.938877105712891 | 1 | false | 1051 | 335 |

配对连续读数：{"E_S_given_D":-1.7389602661132812,"E_remove_with_D":-0.7803306579589844,"E_remove_without_D":-0.23995590209960938,"I_S_D":3.0598316192626953}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
