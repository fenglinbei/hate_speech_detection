# 查询 1160：预测轨迹

主桶：H_residual；focus task：hate。

全部候选标签：["H_residual:hate","Stable_wrong:group"]

来源：{"analysis":"5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77","lexicon":"31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385","plan":"f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f","queries":"0dadeb6de3cc1536aec7862f65bc3611fabdbfba37865a4500b5e28e0ac7dc6b","raw":"505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb"}

## hate

Gold："hate"；四位轨迹：0010；六位轨迹：001000。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | ["non-hate"] | false | 7.694511413574219 | -7.694511413574219 | 1 | false | 119 | 0 |
| CLnew | ["non-hate"] | false | 10.040790557861328 | -10.040790557861328 | 1 | false | 490 | 371 |
| CD | ["hate"] | true | 1.6105918884277344 | 1.6105918884277344 | 1 | false | 664 | 0 |
| CLDnew | ["non-hate"] | false | 2.189380645751953 | -2.189380645751953 | 1 | false | 1035 | 371 |
| CLnewNoCat | ["non-hate"] | false | 9.673316955566406 | -9.673316955566406 | 1 | false | 439 | 320 |
| CLDnewNoCat | ["non-hate"] | false | 0.8663101196289062 | -0.8663101196289062 | 1 | false | 984 | 320 |

配对连续读数：{"E_S_given_D":-2.4769020080566406,"E_S_given_D_hate_logodds":-2.4769020080566406,"E_remove_with_D":1.3230705261230469,"E_remove_with_D_hate_logodds":1.3230705261230469,"E_remove_without_D":0.3674736022949219,"E_remove_without_D_hate_logodds":0.3674736022949219,"I_S_D":-0.4980964660644531,"I_S_D_hate_logodds":-0.4980964660644531}

## group

Gold：["Sexism"]；四位轨迹：0000；六位轨迹：000000。

| condition | prediction | correct | top gap | gold margin | tie count | mode sensitive | prompt tokens | dictionary tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | [] | false | 2.835887908935547 | -6.319797515869141 | 1 | false | 192 | 0 |
| CLnew | [] | false | 0.2471160888671875 | -0.2471160888671875 | 1 | true | 563 | 371 |
| CD | ["Racism"] | false | 2.9698753356933594 | -10.043201446533203 | 1 | false | 751 | 0 |
| CLDnew | ["Racism"] | false | 3.686939239501953 | -12.139209747314453 | 1 | false | 1122 | 371 |
| CLnewNoCat | [] | false | 4.404872894287109 | -4.404872894287109 | 1 | false | 512 | 320 |
| CLDnewNoCat | ["Region"] | false | 6.808811187744141 | -19.4080810546875 | 1 | false | 1071 | 320 |

配对连续读数：{"E_S_given_D":-9.364879608154297,"E_remove_with_D":-7.268871307373047,"E_remove_without_D":-4.157756805419922,"I_S_D":-11.279804229736328}

## 审阅字段

记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。
