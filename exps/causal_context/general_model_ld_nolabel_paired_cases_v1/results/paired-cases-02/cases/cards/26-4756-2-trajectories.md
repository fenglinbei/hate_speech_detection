# 查询 4756：预测轨迹

主桶：H_joint_only；focus task：hate。

全部候选标签：["H_joint_only:hate","Stable_correct:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："hate"；四位轨迹：0001；六位轨迹：000001。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | false | 4.7941741943359375 | -4.7941741943359375 | 1 | false | 124 | 0 |
| CLnew | ["non-hate"] | false | 7.1369476318359375 | -7.1369476318359375 | 1 | false | 349 | 225 |
| CD | ["non-hate"] | false | 0.22957611083984375 | -0.22957611083984375 | 1 | false | 738 | 0 |
| CLDnew | ["non-hate"] | false | 2.7793197631835938 | -2.7793197631835938 | 1 | false | 963 | 225 |
| CLnewNoCat | ["non-hate"] | false | 5.766410827636719 | -5.766410827636719 | 1 | false | 312 | 188 |
| CLDnewNoCat | ["hate"] | true | 0.26149749755859375 | 0.26149749755859375 | 1 | true | 926 | 188 |

配对连续读数：{"E_S_given_D":0.4910736083984375,"E_S_given_D_hate_logodds":0.4910736083984375,"E_remove_with_D":3.0408172607421875,"E_remove_with_D_hate_logodds":3.0408172607421875,"E_remove_without_D":1.3705368041992188,"E_remove_without_D_hate_logodds":1.3705368041992188,"I_S_D":1.4633102416992188,"I_S_D_hate_logodds":1.4633102416992188}

## group

Gold：["Racism"]；四位轨迹：1111；六位轨迹：111111。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["Racism"] | true | 14.933837890625 | 14.933837890625 | 1 | false | 197 | 0 |
| CLnew | ["Racism"] | true | 10.037322998046875 | 10.037322998046875 | 1 | false | 422 | 225 |
| CD | ["Racism"] | true | 19.66889190673828 | 19.66889190673828 | 1 | false | 832 | 0 |
| CLDnew | ["Racism"] | true | 20.01596450805664 | 20.01596450805664 | 1 | false | 1057 | 225 |
| CLnewNoCat | ["Racism"] | true | 9.020694732666016 | 9.020694732666016 | 1 | false | 385 | 188 |
| CLDnewNoCat | ["Racism"] | true | 21.011674880981445 | 21.011674880981445 | 1 | false | 1020 | 188 |

配对连续读数：{"E_S_given_D":1.342782974243164,"E_remove_with_D":0.9957103729248047,"E_remove_without_D":-1.0166282653808594,"I_S_D":7.255926132202148}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
