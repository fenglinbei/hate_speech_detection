# 查询 1746：预测轨迹

主桶：G_category_support；focus task：group。

全部候选标签：["H_residual:hate","G_category_support:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："hate"；四位轨迹：1010；六位轨迹：101000。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["hate"] | true | 0.6219615936279297 | 0.6219615936279297 | 1 | false | 134 | 0 |
| CLnew | ["non-hate"] | false | 2.683116912841797 | -2.683116912841797 | 1 | false | 407 | 273 |
| CD | ["hate"] | true | 3.3735275268554688 | 3.3735275268554688 | 1 | false | 590 | 0 |
| CLDnew | ["non-hate"] | false | 1.1878547668457031 | -1.1878547668457031 | 1 | false | 863 | 273 |
| CLnewNoCat | ["non-hate"] | false | 3.671863555908203 | -3.671863555908203 | 1 | false | 369 | 235 |
| CLDnewNoCat | ["non-hate"] | false | 0.4955463409423828 | -0.4955463409423828 | 1 | false | 825 | 235 |

配对连续读数：{"E_S_given_D":-3.8690738677978516,"E_S_given_D_hate_logodds":-3.8690738677978516,"E_remove_with_D":0.6923084259033203,"E_remove_with_D_hate_logodds":0.6923084259033203,"E_remove_without_D":-0.9887466430664062,"E_remove_without_D_hate_logodds":-0.9887466430664062,"I_S_D":0.42475128173828125,"I_S_D_hate_logodds":0.42475128173828125}

## group

Gold：["Racism","Sexism"]；四位轨迹：0000；六位轨迹：010100。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["Racism"] | false | 5.8580474853515625 | -5.8580474853515625 | 1 | false | 207 | 0 |
| CLnew | ["Racism","Sexism"] | true | 1.9772090911865234 | 1.9772090911865234 | 1 | false | 480 | 273 |
| CD | ["LGBTQ"] | false | 0.046966552734375 | -0.046966552734375 | 1 | true | 678 | 0 |
| CLDnew | ["Racism","Sexism"] | true | 3.4581527709960938 | 3.4581527709960938 | 1 | false | 951 | 273 |
| CLnewNoCat | ["Racism"] | false | 1.4783401489257812 | -1.4783401489257812 | 1 | true | 442 | 235 |
| CLDnewNoCat | ["Racism"] | false | 0.3927459716796875 | -0.3927459716796875 | 1 | true | 913 | 235 |

配对连续读数：{"E_S_given_D":-0.3457794189453125,"E_remove_with_D":-3.8508987426757812,"E_remove_without_D":-3.4555492401123047,"I_S_D":-4.725486755371094}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
