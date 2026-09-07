# 查询 1900：预测轨迹

主桶：H_joint_only；focus task：hate。

全部候选标签：["H_joint_only:hate","Stable_correct:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："hate"；四位轨迹：0001；六位轨迹：000001。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | false | 6.265266418457031 | -6.265266418457031 | 1 | false | 142 | 0 |
| CLnew | ["non-hate"] | false | 3.4533348083496094 | -3.4533348083496094 | 1 | false | 853 | 711 |
| CD | ["non-hate"] | false | 0.976287841796875 | -0.976287841796875 | 1 | false | 993 | 0 |
| CLDnew | ["non-hate"] | false | 0.06050872802734375 | -0.06050872802734375 | 1 | false | 1704 | 711 |
| CLnewNoCat | ["non-hate"] | false | 2.1759300231933594 | -2.1759300231933594 | 1 | false | 761 | 619 |
| CLDnewNoCat | ["hate"] | true | 0.18277740478515625 | 0.18277740478515625 | 1 | true | 1612 | 619 |

配对连续读数：{"E_S_given_D":1.1590652465820312,"E_S_given_D_hate_logodds":1.1590652465820312,"E_remove_with_D":0.2432861328125,"E_remove_with_D_hate_logodds":0.2432861328125,"E_remove_without_D":1.27740478515625,"E_remove_without_D_hate_logodds":1.27740478515625,"I_S_D":-2.9302711486816406,"I_S_D_hate_logodds":-2.9302711486816406}

## group

Gold：["Region"]；四位轨迹：1111；六位轨迹：111111。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["Region"] | true | 7.5943145751953125 | 7.5943145751953125 | 1 | false | 215 | 0 |
| CLnew | ["Region"] | true | 8.72353744506836 | 8.72353744506836 | 1 | false | 926 | 711 |
| CD | ["Region"] | true | 17.065555572509766 | 17.065555572509766 | 1 | false | 1072 | 0 |
| CLDnew | ["Region"] | true | 21.80458641052246 | 21.80458641052246 | 1 | false | 1783 | 711 |
| CLnewNoCat | ["Region"] | true | 8.51681900024414 | 8.51681900024414 | 1 | false | 834 | 619 |
| CLDnewNoCat | ["Region"] | true | 22.169830322265625 | 22.169830322265625 | 1 | false | 1691 | 619 |

配对连续读数：{"E_S_given_D":5.104274749755859,"E_remove_with_D":0.36524391174316406,"E_remove_without_D":-0.20671844482421875,"I_S_D":4.181770324707031}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
