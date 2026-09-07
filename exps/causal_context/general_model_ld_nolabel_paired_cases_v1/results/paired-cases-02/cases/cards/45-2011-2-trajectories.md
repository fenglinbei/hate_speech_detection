# 查询 2011：预测轨迹

主桶：Stable_wrong；focus task：group。

全部候选标签：["H_joint_only:hate","Stable_wrong:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："hate"；四位轨迹：0001；六位轨迹：000001。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | false | 0.22092056274414062 | -0.22092056274414062 | 1 | false | 170 | 0 |
| CLnew | ["non-hate"] | false | 2.9613571166992188 | -2.9613571166992188 | 1 | false | 595 | 425 |
| CD | ["non-hate"] | false | 3.116497039794922 | -3.116497039794922 | 1 | false | 1067 | 0 |
| CLDnew | ["non-hate"] | false | 0.81597900390625 | -0.81597900390625 | 1 | false | 1492 | 425 |
| CLnewNoCat | ["non-hate"] | false | 3.071460723876953 | -3.071460723876953 | 1 | false | 533 | 363 |
| CLDnewNoCat | ["hate"] | true | 0.4302501678466797 | 0.4302501678466797 | 1 | false | 1430 | 363 |

配对连续读数：{"E_S_given_D":3.5467472076416016,"E_S_given_D_hate_logodds":3.5467472076416016,"E_remove_with_D":1.2462291717529297,"E_remove_with_D_hate_logodds":1.2462291717529297,"E_remove_without_D":-0.11010360717773438,"E_remove_without_D_hate_logodds":-0.11010360717773438,"I_S_D":6.397287368774414,"I_S_D_hate_logodds":6.397287368774414}

## group

Gold：["Racism","Region"]；四位轨迹：0000；六位轨迹：000000。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["Racism"] | false | 2.9519309997558594 | -11.026153564453125 | 1 | false | 243 | 0 |
| CLnew | ["Racism"] | false | 0.5463752746582031 | -23.825714111328125 | 1 | true | 668 | 425 |
| CD | ["Racism"] | false | 7.278961181640625 | -19.34539794921875 | 1 | false | 1161 | 0 |
| CLDnew | ["Racism"] | false | 3.102222442626953 | -20.83432388305664 | 1 | false | 1586 | 425 |
| CLnewNoCat | ["Racism"] | false | 1.6200485229492188 | -7.281063079833984 | 1 | true | 606 | 363 |
| CLDnewNoCat | ["Racism"] | false | 12.615913391113281 | -22.569625854492188 | 1 | false | 1524 | 363 |

配对连续读数：{"E_S_given_D":-3.2242279052734375,"E_remove_with_D":-1.7353019714355469,"E_remove_without_D":16.54465103149414,"I_S_D":-6.969318389892578}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
