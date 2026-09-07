# 查询 4189：预测轨迹

主桶：Stable_correct；focus task：group。

全部候选标签：["Stable_correct:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："hate"；四位轨迹：0111；六位轨迹：001111。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | false | 1.5212974548339844 | -1.5212974548339844 | 1 | false | 129 | 0 |
| CLnew | ["non-hate"] | false | 0.1200714111328125 | -0.1200714111328125 | 1 | false | 371 | 242 |
| CD | ["hate"] | true | 3.4381141662597656 | 3.4381141662597656 | 1 | false | 545 | 0 |
| CLDnew | ["hate"] | true | 3.2417984008789062 | 3.2417984008789062 | 1 | false | 787 | 242 |
| CLnewNoCat | ["hate"] | true | 0.6325454711914062 | 0.6325454711914062 | 1 | false | 330 | 201 |
| CLDnewNoCat | ["hate"] | true | 4.251506805419922 | 4.251506805419922 | 1 | false | 746 | 201 |

配对连续读数：{"E_S_given_D":0.8133926391601562,"E_S_given_D_hate_logodds":0.8133926391601562,"E_remove_with_D":1.0097084045410156,"E_remove_with_D_hate_logodds":1.0097084045410156,"E_remove_without_D":0.7526168823242188,"E_remove_without_D_hate_logodds":0.7526168823242188,"I_S_D":-1.3404502868652344,"I_S_D_hate_logodds":-1.3404502868652344}

## group

Gold：["Sexism"]；四位轨迹：1111；六位轨迹：111111。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["Sexism"] | true | 11.391134262084961 | 11.391134262084961 | 1 | false | 202 | 0 |
| CLnew | ["Sexism"] | true | 14.38540267944336 | 14.38540267944336 | 1 | false | 444 | 242 |
| CD | ["Sexism"] | true | 1.5903129577636719 | 1.5903129577636719 | 1 | true | 629 | 0 |
| CLDnew | ["Sexism"] | true | 14.405963897705078 | 14.405963897705078 | 1 | false | 871 | 242 |
| CLnewNoCat | ["Sexism"] | true | 10.488981246948242 | 10.488981246948242 | 1 | false | 403 | 201 |
| CLDnewNoCat | ["Sexism"] | true | 13.556751251220703 | 13.556751251220703 | 1 | false | 830 | 201 |

配对连续读数：{"E_S_given_D":11.966438293457031,"E_remove_with_D":-0.849212646484375,"E_remove_without_D":-3.896421432495117,"I_S_D":12.86859130859375}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
