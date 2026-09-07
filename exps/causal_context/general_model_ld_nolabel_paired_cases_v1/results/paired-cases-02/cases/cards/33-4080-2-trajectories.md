# 查询 4080：预测轨迹

主桶：G_joint_only；focus task：group。

全部候选标签：["G_joint_only:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："hate"；四位轨迹：0011；六位轨迹：001101。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | false | 7.718315124511719 | -7.718315124511719 | 1 | false | 116 | 0 |
| CLnew | ["non-hate"] | false | 6.891170501708984 | -6.891170501708984 | 1 | false | 375 | 259 |
| CD | ["hate"] | true | 2.033519744873047 | 2.033519744873047 | 1 | false | 500 | 0 |
| CLDnew | ["hate"] | true | 1.2726211547851562 | 1.2726211547851562 | 1 | false | 759 | 259 |
| CLnewNoCat | ["non-hate"] | false | 6.08209228515625 | -6.08209228515625 | 1 | false | 340 | 224 |
| CLDnewNoCat | ["hate"] | true | 2.2037734985351562 | 2.2037734985351562 | 1 | false | 724 | 224 |

配对连续读数：{"E_S_given_D":0.17025375366210938,"E_S_given_D_hate_logodds":0.17025375366210938,"E_remove_with_D":0.93115234375,"E_remove_with_D_hate_logodds":0.93115234375,"E_remove_without_D":0.8090782165527344,"E_remove_without_D_hate_logodds":0.8090782165527344,"I_S_D":-1.4659690856933594,"I_S_D_hate_logodds":-1.4659690856933594}

## group

Gold：["Region"]；四位轨迹：0001；六位轨迹：000101。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | [] | false | 5.235139846801758 | -5.235139846801758 | 1 | true | 189 | 0 |
| CLnew | [] | false | 6.0905914306640625 | -6.0905914306640625 | 1 | true | 448 | 259 |
| CD | ["Region","others"] | false | 1.3551788330078125 | -1.3551788330078125 | 1 | false | 575 | 0 |
| CLDnew | ["Region"] | true | 4.83795166015625 | 4.83795166015625 | 1 | false | 834 | 259 |
| CLnewNoCat | [] | false | 1.0926094055175781 | -1.0926094055175781 | 1 | true | 413 | 224 |
| CLDnewNoCat | ["Region"] | true | 1.6655426025390625 | 1.6655426025390625 | 1 | true | 799 | 224 |

配对连续读数：{"E_S_given_D":3.020721435546875,"E_remove_with_D":-3.1724090576171875,"E_remove_without_D":4.997982025146484,"I_S_D":-1.1218090057373047}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
