# 查询 4615：预测轨迹

主桶：H_rescue；focus task：hate。

全部候选标签：["H_rescue:hate","G_category_support:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："hate"；四位轨迹：0011；六位轨迹：001001。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | false | 3.3693695068359375 | -3.3693695068359375 | 1 | false | 125 | 0 |
| CLnew | ["non-hate"] | false | 1.3680801391601562 | -1.3680801391601562 | 1 | false | 524 | 399 |
| CD | ["hate"] | true | 3.434144973754883 | 3.434144973754883 | 1 | false | 678 | 0 |
| CLDnew | ["non-hate"] | false | 0.4918861389160156 | -0.4918861389160156 | 1 | false | 1077 | 399 |
| CLnewNoCat | ["non-hate"] | false | 3.707355499267578 | -3.707355499267578 | 1 | false | 463 | 338 |
| CLDnewNoCat | ["hate"] | true | 0.6347064971923828 | 0.6347064971923828 | 1 | true | 1016 | 338 |

配对连续读数：{"E_S_given_D":-2.7994384765625,"E_S_given_D_hate_logodds":-2.7994384765625,"E_remove_with_D":1.1265926361083984,"E_remove_with_D_hate_logodds":1.1265926361083984,"E_remove_without_D":-2.339275360107422,"E_remove_without_D_hate_logodds":-2.339275360107422,"I_S_D":-2.4614524841308594,"I_S_D_hate_logodds":-2.4614524841308594}

## group

Gold：["Racism","Sexism"]；四位轨迹：0010；六位轨迹：011100。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["Sexism"] | false | 0.6511383056640625 | -16.25863265991211 | 1 | false | 198 | 0 |
| CLnew | ["Racism","Sexism"] | true | 2.544261932373047 | 2.544261932373047 | 1 | false | 597 | 399 |
| CD | ["Racism","Sexism"] | true | 3.1195106506347656 | 3.1195106506347656 | 1 | false | 781 | 0 |
| CLDnew | ["Racism","Sexism"] | true | 15.621101379394531 | 15.621101379394531 | 1 | false | 1180 | 399 |
| CLnewNoCat | ["Racism"] | false | 0.2299346923828125 | -0.743255615234375 | 1 | true | 536 | 338 |
| CLDnewNoCat | ["Racism"] | false | 5.845798492431641 | -5.845798492431641 | 1 | false | 1119 | 338 |

配对连续读数：{"E_S_given_D":-8.965309143066406,"E_remove_with_D":-21.466899871826172,"E_remove_without_D":-3.287517547607422,"I_S_D":-24.48068618774414}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
