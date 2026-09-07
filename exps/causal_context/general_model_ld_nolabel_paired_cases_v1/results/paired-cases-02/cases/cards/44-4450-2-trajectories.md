# 查询 4450：预测轨迹

主桶：Stable_wrong；focus task：hate。

全部候选标签：["Stable_wrong:hate","Stable_correct:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："hate"；四位轨迹：0000；六位轨迹：000000。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | false | 3.6084213256835938 | -3.6084213256835938 | 1 | false | 178 | 0 |
| CLnew | ["non-hate"] | false | 3.7856826782226562 | -3.7856826782226562 | 1 | false | 401 | 223 |
| CD | ["non-hate"] | false | 0.8812370300292969 | -0.8812370300292969 | 1 | false | 917 | 0 |
| CLDnew | ["non-hate"] | false | 1.9671287536621094 | -1.9671287536621094 | 1 | false | 1140 | 223 |
| CLnewNoCat | ["non-hate"] | false | 3.692291259765625 | -3.692291259765625 | 1 | false | 363 | 185 |
| CLDnewNoCat | ["non-hate"] | false | 1.6416435241699219 | -1.6416435241699219 | 1 | false | 1102 | 185 |

配对连续读数：{"E_S_given_D":-0.760406494140625,"E_S_given_D_hate_logodds":-0.760406494140625,"E_remove_with_D":0.3254852294921875,"E_remove_with_D_hate_logodds":0.3254852294921875,"E_remove_without_D":0.09339141845703125,"E_remove_without_D_hate_logodds":0.09339141845703125,"I_S_D":-0.6765365600585938,"I_S_D_hate_logodds":-0.6765365600585938}

## group

Gold：["Region"]；四位轨迹：1111；六位轨迹：111111。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["Region"] | true | 5.117282867431641 | 5.117282867431641 | 1 | false | 251 | 0 |
| CLnew | ["Region"] | true | 10.715953826904297 | 10.715953826904297 | 1 | false | 474 | 223 |
| CD | ["Region"] | true | 7.8732757568359375 | 7.8732757568359375 | 1 | false | 999 | 0 |
| CLDnew | ["Region"] | true | 10.815071105957031 | 10.815071105957031 | 1 | false | 1222 | 223 |
| CLnewNoCat | ["Region"] | true | 10.80731201171875 | 10.80731201171875 | 1 | false | 436 | 185 |
| CLDnewNoCat | ["Region"] | true | 9.68593978881836 | 9.68593978881836 | 1 | false | 1184 | 185 |

配对连续读数：{"E_S_given_D":1.8126640319824219,"E_remove_with_D":-1.1291313171386719,"E_remove_without_D":0.09135818481445312,"I_S_D":-3.8773651123046875}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
