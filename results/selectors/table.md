| arm | causal | physical | budget | logical active unique | physical unique | KL mean | KL 95% CI | PPL ratio | top-1 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| teacher-dense | unavailable | non_compliant | noncompliant-dense | 1720574976 | 1720574976 | 0.000000 | [0.000000, 0.000000] | 1 | 1.0000 |
| masked-all | available | non_compliant | noncompliant-all-groups | 1720574976 | 1720574976 | 0.000000 | [0.000000, 0.000000] | 1 | 1.0000 |
| token-magnitude-sparse | available | non_compliant | B142 | 998630400 | 1479926784 | 6.676772 | [6.541269, 6.813145] | 457.211 | 0.0706 |
| token-magnitude | available | non_compliant | B142 | 998630400 | 1479926784 | 6.676772 | [6.541269, 6.813145] | 457.212 | 0.0706 |
| token-frequency-sparse | available | non_compliant | B142 | 998630400 | 1479926784 | 7.034329 | [6.889674, 7.182299] | 696.72 | 0.0616 |
| token-frequency | available | non_compliant | B142 | 998630400 | 1479926784 | 7.034329 | [6.889674, 7.182299] | 696.72 | 0.0616 |
| static-greedy-uniform5 | available | non_compliant | B142 | 998630400 | 998630400 | 7.706657 | [7.512359, 7.922955] | 1196.79 | 0.0239 |
| static-greedy-uniform5-sparse | available | compliant | B142 | 998630400 | 998630400 | 7.706658 | [7.512359, 7.922955] | 1196.79 | 0.0239 |
| token-task-static-nn | unavailable | non_compliant | B142 | 998630400 | 998630400 | 9.580137 | [9.284461, 9.897374] | 7896.43 | 0.0095 |
| static-greedy-swap4-sparse | available | compliant | M1-K3-112 | 927851520 | 927851520 | 9.895816 | [9.643577, 10.153090] | 11033.7 | 0.0081 |
| static-greedy-swap4 | available | non_compliant | M1-K3-112 | 927851520 | 927851520 | 9.895818 | [9.643579, 10.153094] | 11033.7 | 0.0081 |
| static-magnitude-k5-sparse | available | compliant | B140 | 993911808 | 993911808 | 10.019480 | [9.779426, 10.270834] | 11772.9 | 0.0080 |
| static-magnitude-k5 | available | non_compliant | B140 | 993911808 | 993911808 | 10.019482 | [9.779430, 10.270835] | 11772.9 | 0.0080 |
| static-frequency-k5 | available | non_compliant | B140 | 993911808 | 993911808 | 10.309631 | [10.053809, 10.574238] | 16580 | 0.0085 |
| static-frequency-k5-sparse | available | compliant | B140 | 993911808 | 993911808 | 10.309633 | [10.053808, 10.574243] | 16580 | 0.0085 |
| token-learned-unary | available | non_compliant | B142 | 998631268 | 1479927652 | 10.474509 | [10.180010, 10.792109] | 18318.3 | 0.0074 |
| token-learned-interaction | available | non_compliant | B142 | 998631268 | 1479927652 | 10.482465 | [10.240315, 10.734217] | 18070.7 | 0.0072 |
| token-random-sparse | available | compliant | B142 | 998630400 | 998630400 | 10.745491 | [10.447647, 11.025083] | 27897.5 | 0.0111 |
| token-random | available | non_compliant | B142 | 998630400 | 998630400 | 10.745493 | [10.447650, 11.025086] | 27897.6 | 0.0111 |
| token-gxo | unavailable | non_compliant | B142 | 998630400 | 1479926784 | 11.137447 | [10.902189, 11.376774] | 58540.8 | 0.0101 |
| token-cor-gxo | unavailable | non_compliant | B142 | 998630400 | 1479926784 | 11.322907 | [11.028465, 11.641592] | 75719.9 | 0.0111 |
| static-random-k5-sparse | available | compliant | B140 | 993911808 | 993911808 | 11.782509 | [11.507477, 12.071168] | 57418.1 | 0.0015 |
| static-random-k5 | available | non_compliant | B140 | 993911808 | 993911808 | 11.782510 | [11.507479, 12.071168] | 57418.1 | 0.0015 |

Oracle (causal unavailable, physical non-compliant):

| arm | budget | KL mean | KL 95% CI | PPL ratio | top-1 | frac KL<=0.05 | frac KL<=0.25 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| oracle-local-64 | G64 | 3.665418 | [3.289385, 4.034953] | 25.4515 | 0.2693 | 0.0169 | 0.0550 |
| oracle-final-refine | B142 | 5.785897 | [5.159938, 6.411855] | 177.664 | 0.1018 | 0.0000 | 0.0029 |
| oracle-local-16-B142 | B142 | 5.952733 | [5.644695, 6.224299] | 229.896 | 0.0866 | 0.0005 | 0.0046 |
| oracle-local-16-B140 | B140 | 6.273271 | [5.980226, 6.535457] | 327.762 | 0.0697 | 0.0005 | 0.0042 |
| oracle-local-4 | G4 | 7.833459 | [7.600261, 8.046133] | 1550.59 | 0.0560 | 0.0000 | 0.0000 |
