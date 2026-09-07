# 查询 7244：预测轨迹

主桶：G_category_support；focus task：group。

全部候选标签：["H_residual:hate","G_category_support:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："hate"；四位轨迹：0010；六位轨迹：001000。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | false | 3.7614059448242188 | -3.7614059448242188 | 1 | false | 149 | 0 |
| CLnew | ["non-hate"] | false | 5.556243896484375 | -5.556243896484375 | 1 | false | 839 | 690 |
| CD | ["hate"] | true | 2.4041099548339844 | 2.4041099548339844 | 1 | false | 583 | 0 |
| CLDnew | ["non-hate"] | false | 1.8127098083496094 | -1.8127098083496094 | 1 | false | 1273 | 690 |
| CLnewNoCat | ["non-hate"] | false | 2.1442337036132812 | -2.1442337036132812 | 1 | false | 750 | 601 |
| CLDnewNoCat | ["non-hate"] | false | 4.94537353515625 | -4.94537353515625 | 1 | false | 1184 | 601 |

配对连续读数：{"E_S_given_D":-7.349483489990234,"E_S_given_D_hate_logodds":-7.349483489990234,"E_remove_with_D":-3.1326637268066406,"E_remove_with_D_hate_logodds":-3.1326637268066406,"E_remove_without_D":3.4120101928710938,"E_remove_without_D_hate_logodds":3.4120101928710938,"I_S_D":-8.966655731201172,"I_S_D_hate_logodds":-8.966655731201172}

## group

Gold：["Region"]；四位轨迹：1001；六位轨迹：110101。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["Region"] | true | 2.839153289794922 | 2.839153289794922 | 1 | true | 222 | 0 |
| CLnew | ["Region"] | true | 7.512775421142578 | 7.512775421142578 | 1 | false | 912 | 690 |
| CD | ["Region","Sexism"] | false | 0.42200469970703125 | -0.42200469970703125 | 1 | false | 661 | 0 |
| CLDnew | ["Region"] | true | 21.254432678222656 | 21.254432678222656 | 1 | false | 1351 | 690 |
| CLnewNoCat | [] | false | 2.847156524658203 | -5.650386810302734 | 1 | true | 823 | 601 |
| CLDnewNoCat | ["Region"] | true | 13.591114044189453 | 13.591114044189453 | 1 | false | 1262 | 601 |

配对连续读数：{"E_S_given_D":14.013118743896484,"E_remove_with_D":-7.663318634033203,"E_remove_without_D":-13.163162231445312,"I_S_D":22.50265884399414}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
