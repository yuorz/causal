### ames
```bash
python cli.py split --dataset ./data/ames/AmesHousing.csv   --train-name AmesHousing_train.csv --valid-name AmesHousing_valid.csv
python cli.py train --dataset ./data/ames/AmesHousing.csv   --task regression   --target SalePrice   --exclude Order PID   --method random   --ratio 1.0   --model linear
```

### housing
```bash
python cli.py split --dataset ./data/housing/Housing.csv   --train-name Housing_train.csv --valid-name Housing_valid.csv
python cli.py train --dataset ./data/housing/Housing.csv   --task regression   --target price --method random   --ratio 1.0   --model mlp
```

### covid
```bash
python cli.py split --dataset ./data/covid.csv --train-name covid_train.csv --valid-name covid_valid.csv
python cli.py train --dataset ./data/covid.csv --task regression --target tested_positive.2 --exclude id AL AZ CA CO CT FL GA IL IN IA KS KY LA ME MD MA MI MN MO NJ NM NY NC OH OK OR PA SC TN TX VA WA WV WI --method random --ratio 0.7
```

### [bank](https://archive.ics.uci.edu/dataset/222/bank+marketing)
```bash
python cli.py split --dataset ./data/bank/bank-full.csv --train-name bank-full_train.csv --valid-name bank-full_valid.csv
python cli.py train --dataset ./data/bank/bank-full.csv --task classification --target y --method random --ratio 0.4 --model mlp
```

### winequality-white
```bash
python cli.py split --dataset ./data/wine/winequality-white.csv --train-name winequality-white_train.csv --valid-name winequality-white_valid.csv
python cli.py train --dataset ./data/wine/winequality-white.csv --task regression --target quality --method random --ratio 1.0 --model linear
```