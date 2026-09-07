# 查询 2297：预测轨迹

主桶：H_joint_only；focus task：hate。

全部候选标签：["H_joint_only:hate","G_joint_only:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："hate"；四位轨迹：0001；六位轨迹：000001。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | false | 4.697978973388672 | -4.697978973388672 | 1 | false | 181 | 0 |
| CLnew | ["non-hate"] | false | 6.7543487548828125 | -6.7543487548828125 | 1 | false | 494 | 313 |
| CD | ["non-hate"] | false | 0.6112995147705078 | -0.6112995147705078 | 1 | false | 1351 | 0 |
| CLDnew | ["non-hate"] | false | 0.6443595886230469 | -0.6443595886230469 | 1 | false | 1664 | 313 |
| CLnewNoCat | ["non-hate"] | false | 7.106296539306641 | -7.106296539306641 | 1 | false | 457 | 276 |
| CLDnewNoCat | ["hate"] | true | 0.4340629577636719 | 0.4340629577636719 | 1 | false | 1627 | 276 |

配对连续读数：{"E_S_given_D":1.0453624725341797,"E_S_given_D_hate_logodds":1.0453624725341797,"E_remove_with_D":1.0784225463867188,"E_remove_with_D_hate_logodds":1.0784225463867188,"E_remove_without_D":-0.3519477844238281,"E_remove_without_D_hate_logodds":-0.3519477844238281,"I_S_D":3.4536800384521484,"I_S_D_hate_logodds":3.4536800384521484}

## group

Gold：["Region"]；四位轨迹：0001；六位轨迹：000001。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | [] | false | 0.9841232299804688 | -6.877124786376953 | 1 | true | 254 | 0 |
| CLnew | [] | false | 1.2672615051269531 | -17.171287536621094 | 1 | false | 567 | 313 |
| CD | ["Region","others"] | false | 3.6072998046875 | -8.40548324584961 | 1 | false | 1434 | 0 |
| CLDnew | ["Region","others"] | false | 0.3410682678222656 | -0.3410682678222656 | 1 | false | 1747 | 313 |
| CLnewNoCat | [] | false | 1.5027542114257812 | -15.446491241455078 | 1 | false | 530 | 276 |
| CLDnewNoCat | ["Region"] | true | 1.3415565490722656 | 1.3415565490722656 | 1 | false | 1710 | 276 |

配对连续读数：{"E_S_given_D":9.747039794921875,"E_remove_with_D":1.6826248168945312,"E_remove_without_D":1.7247962951660156,"I_S_D":18.31640625}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
