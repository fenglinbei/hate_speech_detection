# 查询 3683：预测轨迹

主桶：H_rescue；focus task：hate。

全部候选标签：["H_rescue:hate","G_category_harm:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："hate"；四位轨迹：0011；六位轨迹：001001。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | false | 3.9400291442871094 | -3.9400291442871094 | 1 | false | 118 | 0 |
| CLnew | ["non-hate"] | false | 12.556705474853516 | -12.556705474853516 | 1 | false | 247 | 129 |
| CD | ["hate"] | true | 3.166330337524414 | 3.166330337524414 | 1 | false | 560 | 0 |
| CLDnew | ["non-hate"] | false | 0.5390129089355469 | -0.5390129089355469 | 1 | false | 689 | 129 |
| CLnewNoCat | ["non-hate"] | false | 12.106021881103516 | -12.106021881103516 | 1 | false | 226 | 108 |
| CLDnewNoCat | ["hate"] | true | 0.7276153564453125 | 0.7276153564453125 | 1 | false | 668 | 108 |

配对连续读数：{"E_S_given_D":-2.4387149810791016,"E_S_given_D_hate_logodds":-2.4387149810791016,"E_remove_with_D":1.2666282653808594,"E_remove_with_D_hate_logodds":1.2666282653808594,"E_remove_without_D":0.45068359375,"E_remove_without_D_hate_logodds":0.45068359375,"I_S_D":5.727277755737305,"I_S_D_hate_logodds":5.727277755737305}

## group

Gold：["Region"]；四位轨迹：1111；六位轨迹：101111。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["Region"] | true | 6.816566467285156 | 6.816566467285156 | 1 | false | 191 | 0 |
| CLnew | [] | false | 2.3984603881835938 | -2.3984603881835938 | 1 | false | 320 | 129 |
| CD | ["Region"] | true | 13.158893585205078 | 13.158893585205078 | 1 | false | 640 | 0 |
| CLDnew | ["Region"] | true | 9.05633544921875 | 9.05633544921875 | 1 | false | 769 | 129 |
| CLnewNoCat | ["Region"] | true | 0.7756233215332031 | 0.7756233215332031 | 1 | false | 299 | 108 |
| CLDnewNoCat | ["Region"] | true | 10.197887420654297 | 10.197887420654297 | 1 | false | 748 | 108 |

配对连续读数：{"E_S_given_D":-2.9610061645507812,"E_remove_with_D":1.1415519714355469,"E_remove_without_D":3.174083709716797,"I_S_D":3.079936981201172}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
