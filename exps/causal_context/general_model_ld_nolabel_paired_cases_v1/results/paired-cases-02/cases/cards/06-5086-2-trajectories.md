# 查询 5086：预测轨迹

主桶：H_residual；focus task：hate。

全部候选标签：["H_residual:hate"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："hate"；四位轨迹：0110；六位轨迹：001010。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | false | 2.3008174896240234 | -2.3008174896240234 | 1 | false | 149 | 0 |
| CLnew | ["non-hate"] | false | 0.15942764282226562 | -0.15942764282226562 | 1 | false | 664 | 515 |
| CD | ["hate"] | true | 2.634798049926758 | 2.634798049926758 | 1 | false | 831 | 0 |
| CLDnew | ["non-hate"] | false | 0.24813079833984375 | -0.24813079833984375 | 1 | false | 1346 | 515 |
| CLnewNoCat | ["hate"] | true | 1.2169609069824219 | 1.2169609069824219 | 1 | false | 564 | 415 |
| CLDnewNoCat | ["non-hate"] | false | 0.277862548828125 | -0.277862548828125 | 1 | false | 1246 | 415 |

配对连续读数：{"E_S_given_D":-2.912660598754883,"E_S_given_D_hate_logodds":-2.912660598754883,"E_remove_with_D":-0.02973175048828125,"E_remove_with_D_hate_logodds":-0.02973175048828125,"E_remove_without_D":1.3763885498046875,"E_remove_without_D_hate_logodds":1.3763885498046875,"I_S_D":-6.430438995361328,"I_S_D_hate_logodds":-6.430438995361328}

## group

Gold：["LGBTQ","others"]；四位轨迹：0011；六位轨迹：001101。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["LGBTQ"] | false | 10.784156799316406 | -25.100627899169922 | 1 | false | 222 | 0 |
| CLnew | ["LGBTQ"] | false | 7.254142761230469 | -7.254142761230469 | 1 | false | 737 | 515 |
| CD | ["LGBTQ","others"] | true | 6.848213195800781 | 6.848213195800781 | 1 | false | 932 | 0 |
| CLDnew | ["LGBTQ","others"] | true | 20.071430206298828 | 20.071430206298828 | 1 | false | 1447 | 515 |
| CLnewNoCat | ["LGBTQ"] | false | 5.288166046142578 | -20.499202728271484 | 1 | false | 637 | 415 |
| CLDnewNoCat | ["LGBTQ","others"] | true | 7.807151794433594 | 7.807151794433594 | 1 | false | 1347 | 415 |

配对连续读数：{"E_S_given_D":0.9589385986328125,"E_remove_with_D":-12.264278411865234,"E_remove_without_D":-13.245059967041016,"I_S_D":-3.642486572265625}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
