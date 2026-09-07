# 查询 1292：预测轨迹

主桶：H_removal_harm；focus task：hate。

全部候选标签：["H_removal_harm:hate","Stable_wrong:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："non-hate"；四位轨迹：1100；六位轨迹：110110。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | true | 3.1864662170410156 | 3.1864662170410156 | 1 | false | 114 | 0 |
| CLnew | ["non-hate"] | true | 2.592945098876953 | 2.592945098876953 | 1 | false | 301 | 187 |
| CD | ["hate"] | false | 3.1349411010742188 | -3.1349411010742188 | 1 | false | 470 | 0 |
| CLDnew | ["non-hate"] | true | 2.1092300415039062 | 2.1092300415039062 | 1 | false | 657 | 187 |
| CLnewNoCat | ["non-hate"] | true | 3.310546875 | 3.310546875 | 1 | false | 281 | 167 |
| CLDnewNoCat | ["hate"] | false | 0.30350494384765625 | -0.30350494384765625 | 1 | true | 637 | 167 |

配对连续读数：{"E_S_given_D":2.8314361572265625,"E_S_given_D_hate_logodds":-2.8314361572265625,"E_remove_with_D":-2.4127349853515625,"E_remove_with_D_hate_logodds":2.4127349853515625,"E_remove_without_D":0.7176017761230469,"E_remove_without_D_hate_logodds":-0.7176017761230469,"I_S_D":2.707355499267578,"I_S_D_hate_logodds":-2.707355499267578}

## group

Gold：[]；四位轨迹：0000；六位轨迹：000000。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["Racism"] | false | 14.749235153198242 | -14.749235153198242 | 1 | false | 187 | 0 |
| CLnew | ["Racism"] | false | 14.980705261230469 | -14.980705261230469 | 1 | false | 374 | 187 |
| CD | ["Racism"] | false | 15.980209350585938 | -19.34498405456543 | 1 | false | 551 | 0 |
| CLDnew | ["Racism"] | false | 20.704689025878906 | -21.137863159179688 | 1 | false | 738 | 187 |
| CLnewNoCat | ["Racism"] | false | 13.304378509521484 | -13.304378509521484 | 1 | false | 354 | 167 |
| CLDnewNoCat | ["Racism"] | false | 21.094467163085938 | -22.508569717407227 | 1 | false | 718 | 167 |

配对连续读数：{"E_S_given_D":-3.163585662841797,"E_remove_with_D":-1.370706558227539,"E_remove_without_D":1.6763267517089844,"I_S_D":-4.608442306518555}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
