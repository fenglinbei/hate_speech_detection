# 查询 4026：预测轨迹

主桶：Stable_wrong；focus task：hate。

全部候选标签：["Stable_wrong:hate","Stable_correct:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："hate"；四位轨迹：0000；六位轨迹：000000。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | false | 6.5742645263671875 | -6.5742645263671875 | 1 | false | 148 | 0 |
| CLnew | ["non-hate"] | false | 8.230133056640625 | -8.230133056640625 | 1 | false | 360 | 212 |
| CD | ["non-hate"] | false | 0.8593330383300781 | -0.8593330383300781 | 1 | false | 974 | 0 |
| CLDnew | ["non-hate"] | false | 3.341930389404297 | -3.341930389404297 | 1 | false | 1186 | 212 |
| CLnewNoCat | ["non-hate"] | false | 8.712928771972656 | -8.712928771972656 | 1 | false | 331 | 183 |
| CLDnewNoCat | ["non-hate"] | false | 1.4697036743164062 | -1.4697036743164062 | 1 | false | 1157 | 183 |

配对连续读数：{"E_S_given_D":-0.6103706359863281,"E_S_given_D_hate_logodds":-0.6103706359863281,"E_remove_with_D":1.8722267150878906,"E_remove_with_D_hate_logodds":1.8722267150878906,"E_remove_without_D":-0.48279571533203125,"E_remove_without_D_hate_logodds":-0.48279571533203125,"I_S_D":1.5282936096191406,"I_S_D_hate_logodds":1.5282936096191406}

## group

Gold：["Region"]；四位轨迹：1111；六位轨迹：111111。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["Region"] | true | 10.26962661743164 | 10.26962661743164 | 1 | false | 221 | 0 |
| CLnew | ["Region"] | true | 6.292934417724609 | 6.292934417724609 | 1 | false | 433 | 212 |
| CD | ["Region"] | true | 9.616138458251953 | 9.616138458251953 | 1 | false | 1059 | 0 |
| CLDnew | ["Region"] | true | 3.751300811767578 | 3.751300811767578 | 1 | false | 1271 | 212 |
| CLnewNoCat | ["Region"] | true | 6.3099365234375 | 6.3099365234375 | 1 | false | 404 | 183 |
| CLDnewNoCat | ["Region"] | true | 12.255767822265625 | 12.255767822265625 | 1 | false | 1242 | 183 |

配对连续读数：{"E_S_given_D":2.639629364013672,"E_remove_with_D":8.504467010498047,"E_remove_without_D":0.017002105712890625,"I_S_D":6.5993194580078125}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
