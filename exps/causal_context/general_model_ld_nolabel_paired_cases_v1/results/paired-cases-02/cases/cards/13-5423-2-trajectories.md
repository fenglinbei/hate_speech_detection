# 查询 5423：预测轨迹

主桶：H_removal_harm；focus task：hate。

全部候选标签：["H_removal_harm:hate"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："hate"；四位轨迹：0010；六位轨迹：001100。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | false | 12.093570709228516 | -12.093570709228516 | 1 | false | 114 | 0 |
| CLnew | ["non-hate"] | false | 7.5418548583984375 | -7.5418548583984375 | 1 | false | 435 | 320 |
| CD | ["hate"] | true | 3.0232620239257812 | 3.0232620239257812 | 1 | false | 412 | 0 |
| CLDnew | ["hate"] | true | 0.13299942016601562 | 0.13299942016601562 | 1 | true | 733 | 320 |
| CLnewNoCat | ["non-hate"] | false | 5.097103118896484 | -5.097103118896484 | 1 | false | 398 | 283 |
| CLDnewNoCat | ["non-hate"] | false | 1.1964836120605469 | -1.1964836120605469 | 1 | false | 696 | 283 |

配对连续读数：{"E_S_given_D":-4.219745635986328,"E_S_given_D_hate_logodds":-4.219745635986328,"E_remove_with_D":-1.3294830322265625,"E_remove_with_D_hate_logodds":-1.3294830322265625,"E_remove_without_D":2.444751739501953,"E_remove_without_D_hate_logodds":2.444751739501953,"I_S_D":-11.21621322631836,"I_S_D_hate_logodds":-11.21621322631836}

## group

Gold：["Racism"]；四位轨迹：0111；六位轨迹：011111。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | [] | false | 18.437084197998047 | -27.798925399780273 | 1 | false | 187 | 0 |
| CLnew | ["Racism"] | true | 4.791507720947266 | 4.791507720947266 | 1 | false | 508 | 320 |
| CD | ["Racism"] | true | 1.4626235961914062 | 1.4626235961914062 | 1 | false | 491 | 0 |
| CLDnew | ["Racism"] | true | 18.41189956665039 | 18.41189956665039 | 1 | false | 812 | 320 |
| CLnewNoCat | ["Racism"] | true | 5.904792785644531 | 5.904792785644531 | 1 | false | 471 | 283 |
| CLDnewNoCat | ["Racism"] | true | 18.50466537475586 | 18.50466537475586 | 1 | false | 775 | 283 |

配对连续读数：{"E_S_given_D":17.042041778564453,"E_remove_with_D":0.09276580810546875,"E_remove_without_D":1.1132850646972656,"I_S_D":-16.66167640686035}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
