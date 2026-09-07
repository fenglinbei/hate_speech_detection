# 查询 2802：预测轨迹

主桶：G_joint_only；focus task：group。

全部候选标签：["H_rescue:hate","G_category_support:group","G_joint_only:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："hate"；四位轨迹：0011；六位轨迹：001001。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | false | 1.2924118041992188 | -1.2924118041992188 | 1 | false | 116 | 0 |
| CLnew | ["non-hate"] | false | 1.2004013061523438 | -1.2004013061523438 | 1 | false | 633 | 517 |
| CD | ["hate"] | true | 2.250091552734375 | 2.250091552734375 | 1 | false | 443 | 0 |
| CLDnew | ["non-hate"] | false | 1.8384361267089844 | -1.8384361267089844 | 1 | false | 960 | 517 |
| CLnewNoCat | ["non-hate"] | false | 0.8973808288574219 | -0.8973808288574219 | 1 | false | 560 | 444 |
| CLDnewNoCat | ["hate"] | true | 0.9201412200927734 | 0.9201412200927734 | 1 | true | 887 | 444 |

配对连续读数：{"E_S_given_D":-1.3299503326416016,"E_S_given_D_hate_logodds":-1.3299503326416016,"E_remove_with_D":2.758577346801758,"E_remove_with_D_hate_logodds":2.758577346801758,"E_remove_without_D":0.3030204772949219,"E_remove_without_D_hate_logodds":0.3030204772949219,"I_S_D":-1.7249813079833984,"I_S_D_hate_logodds":-1.7249813079833984}

## group

Gold：["Racism","Sexism"]；四位轨迹：0001；六位轨迹：010101。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["LGBTQ"] | false | 10.954734802246094 | -27.75302505493164 | 1 | false | 189 | 0 |
| CLnew | ["Racism","Sexism"] | true | 1.5704212188720703 | 1.5704212188720703 | 1 | false | 706 | 517 |
| CD | ["LGBTQ"] | false | 3.785552978515625 | -12.339069366455078 | 1 | false | 517 | 0 |
| CLDnew | ["Racism","Sexism"] | true | 11.326265335083008 | 11.326265335083008 | 1 | false | 1034 | 517 |
| CLnewNoCat | ["LGBTQ","Sexism"] | false | 0.9432182312011719 | -15.738521575927734 | 1 | true | 633 | 444 |
| CLDnewNoCat | ["Racism","Sexism"] | true | 10.494693756103516 | 10.494693756103516 | 1 | false | 961 | 444 |

配对连续读数：{"E_S_given_D":22.833763122558594,"E_remove_with_D":-0.8315715789794922,"E_remove_without_D":-17.308942794799805,"I_S_D":10.819259643554688}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
