"""Model comparison by scaffold cross-validation and the final holdout evaluation."""

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES, N_FOLDS, TARGET
from src.data import load_dataset
from src.evaluate import format_report, report
from src.splits import scaffold_groups, scaffold_split


def cv_report(model, train):
    groups = scaffold_groups(train)
    pred = cross_val_predict(model, train[FEATURES], train[TARGET], groups=groups, cv=GroupKFold(N_FOLDS))
    return report(train[TARGET], pred)


def main():
    df = load_dataset()
    train, holdout = scaffold_split(df)

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    print(format_report("Ridge CV", cv_report(ridge, train)))
    random_cv = cross_val_predict(ridge, train[FEATURES], train[TARGET], cv=5)
    print(format_report("Ridge CV (5 folds)", report(train[TARGET], random_cv)))

    final = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    final.fit(train[FEATURES], train[TARGET])
    print(format_report("Ridge holdout", report(holdout[TARGET], final.predict(holdout[FEATURES]))))


if __name__ == "__main__":
    main()
