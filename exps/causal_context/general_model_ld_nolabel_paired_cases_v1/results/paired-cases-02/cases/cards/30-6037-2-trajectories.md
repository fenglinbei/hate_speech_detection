# 查询 6037：预测轨迹

主桶：G_joint_only；focus task：group。

全部候选标签：["Stable_wrong:hate","G_joint_only:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："hate"；四位轨迹：0000；六位轨迹：000000。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | false | 14.739086151123047 | -14.739086151123047 | 1 | false | 115 | 0 |
| CLnew | ["non-hate"] | false | 10.897315979003906 | -10.897315979003906 | 1 | false | 244 | 129 |
| CD | ["non-hate"] | false | 11.75667953491211 | -11.75667953491211 | 1 | false | 435 | 0 |
| CLDnew | ["non-hate"] | false | 3.8335723876953125 | -3.8335723876953125 | 1 | false | 564 | 129 |
| CLnewNoCat | ["non-hate"] | false | 9.830673217773438 | -9.830673217773438 | 1 | false | 227 | 112 |
| CLDnewNoCat | ["non-hate"] | false | 7.159782409667969 | -7.159782409667969 | 1 | false | 547 | 112 |

配对连续读数：{"E_S_given_D":4.596897125244141,"E_S_given_D_hate_logodds":4.596897125244141,"E_remove_with_D":-3.3262100219726562,"E_remove_with_D_hate_logodds":-3.3262100219726562,"E_remove_without_D":1.0666427612304688,"E_remove_without_D_hate_logodds":1.0666427612304688,"I_S_D":-0.31151580810546875,"I_S_D_hate_logodds":-0.31151580810546875}

## group

Gold：["Sexism"]；四位轨迹：0001；六位轨迹：000101。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | [] | false | 19.829639434814453 | -31.29572296142578 | 1 | false | 188 | 0 |
| CLnew | [] | false | 6.890178680419922 | -6.890178680419922 | 1 | false | 317 | 129 |
| CD | [] | false | 12.682819366455078 | -12.682819366455078 | 1 | false | 509 | 0 |
| CLDnew | ["Sexism"] | true | 5.678356170654297 | 5.678356170654297 | 1 | false | 638 | 129 |
| CLnewNoCat | [] | false | 1.5585250854492188 | -1.5585250854492188 | 1 | false | 300 | 112 |
| CLDnewNoCat | ["Sexism"] | true | 2.3382186889648438 | 2.3382186889648438 | 1 | false | 621 | 112 |

配对连续读数：{"E_S_given_D":15.021038055419922,"E_remove_with_D":-3.340137481689453,"E_remove_without_D":5.331653594970703,"I_S_D":-14.71615982055664}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
