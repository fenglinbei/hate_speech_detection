# 查询 4137：预测轨迹

主桶：Stable_wrong；focus task：group。

全部候选标签：["Stable_correct:hate","Stable_wrong:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："non-hate"；四位轨迹：1111；六位轨迹：111111。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | true | 6.4754486083984375 | 6.4754486083984375 | 1 | false | 110 | 0 |
| CLnew | ["non-hate"] | true | 6.941196441650391 | 6.941196441650391 | 1 | false | 280 | 170 |
| CD | ["non-hate"] | true | 1.654144287109375 | 1.654144287109375 | 1 | false | 463 | 0 |
| CLDnew | ["non-hate"] | true | 2.4750289916992188 | 2.4750289916992188 | 1 | false | 633 | 170 |
| CLnewNoCat | ["non-hate"] | true | 6.304409027099609 | 6.304409027099609 | 1 | false | 253 | 143 |
| CLDnewNoCat | ["non-hate"] | true | 0.9610366821289062 | 0.9610366821289062 | 1 | false | 606 | 143 |

配对连续读数：{"E_S_given_D":-0.6931076049804688,"E_S_given_D_hate_logodds":0.6931076049804688,"E_remove_with_D":-1.5139923095703125,"E_remove_with_D_hate_logodds":1.5139923095703125,"E_remove_without_D":-0.6367874145507812,"E_remove_without_D_hate_logodds":0.6367874145507812,"I_S_D":-0.5220680236816406,"I_S_D_hate_logodds":0.5220680236816406}

## group

Gold：[]；四位轨迹：0000；六位轨迹：000000。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["Sexism"] | false | 13.508487701416016 | -13.543060302734375 | 1 | false | 183 | 0 |
| CLnew | ["LGBTQ"] | false | 7.461055755615234 | -7.461055755615234 | 1 | false | 353 | 170 |
| CD | ["LGBTQ"] | false | 22.202383041381836 | -22.202383041381836 | 1 | false | 541 | 0 |
| CLDnew | ["LGBTQ"] | false | 18.135250091552734 | -21.11278533935547 | 1 | false | 711 | 170 |
| CLnewNoCat | ["LGBTQ"] | false | 2.4982986450195312 | -3.746898651123047 | 1 | false | 326 | 143 |
| CLDnewNoCat | ["LGBTQ"] | false | 21.102928161621094 | -21.102928161621094 | 1 | false | 684 | 143 |

配对连续读数：{"E_S_given_D":1.0994548797607422,"E_remove_with_D":0.009857177734375,"E_remove_without_D":3.7141571044921875,"I_S_D":-8.696706771850586}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
