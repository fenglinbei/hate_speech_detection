# 查询 376：预测轨迹

主桶：G_joint_only；focus task：group。

全部候选标签：["Stable_correct:hate","G_joint_only:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："hate"；四位轨迹：1111；六位轨迹：111111。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["hate"] | true | 6.068571090698242 | 6.068571090698242 | 1 | false | 152 | 0 |
| CLnew | ["hate"] | true | 0.831787109375 | 0.831787109375 | 1 | false | 398 | 246 |
| CD | ["hate"] | true | 7.888019561767578 | 7.888019561767578 | 1 | false | 875 | 0 |
| CLDnew | ["hate"] | true | 4.563285827636719 | 4.563285827636719 | 1 | false | 1121 | 246 |
| CLnewNoCat | ["hate"] | true | 3.4446258544921875 | 3.4446258544921875 | 1 | false | 355 | 203 |
| CLDnewNoCat | ["hate"] | true | 4.416873931884766 | 4.416873931884766 | 1 | false | 1078 | 203 |

配对连续读数：{"E_S_given_D":-3.4711456298828125,"E_S_given_D_hate_logodds":-3.4711456298828125,"E_remove_with_D":-0.14641189575195312,"E_remove_with_D_hate_logodds":-0.14641189575195312,"E_remove_without_D":2.6128387451171875,"E_remove_without_D_hate_logodds":2.6128387451171875,"I_S_D":-0.8472003936767578,"I_S_D_hate_logodds":-0.8472003936767578}

## group

Gold：["LGBTQ","others"]；四位轨迹：0001；六位轨迹：000101。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["LGBTQ","Sexism"] | false | 1.4918594360351562 | -15.798633575439453 | 1 | false | 225 | 0 |
| CLnew | ["LGBTQ","Sexism"] | false | 0.8471107482910156 | -0.8471107482910156 | 1 | false | 471 | 246 |
| CD | ["LGBTQ","Sexism"] | false | 0.7788925170898438 | -0.7788925170898438 | 1 | false | 974 | 0 |
| CLDnew | ["LGBTQ","others"] | true | 14.923011779785156 | 14.923011779785156 | 1 | false | 1220 | 246 |
| CLnewNoCat | ["LGBTQ","Sexism"] | false | 4.541694641113281 | -12.315353393554688 | 1 | false | 428 | 203 |
| CLDnewNoCat | ["LGBTQ","others"] | true | 7.101413726806641 | 7.101413726806641 | 1 | false | 1177 | 203 |

配对连续读数：{"E_S_given_D":7.880306243896484,"E_remove_with_D":-7.821598052978516,"E_remove_without_D":-11.468242645263672,"I_S_D":4.397026062011719}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
