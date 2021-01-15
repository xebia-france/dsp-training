from enum import Enum


class Loans(str, Enum):
    Loan_Status = "Loan_Status"
    Credit_History = "Credit_History"
    Self_Employed = "Self_Employed"
    LoanAmount = "LoanAmount"
    Gender = "Gender"
    Dependents = "Dependents"
    Loan_Amount_Term = "Loan_Amount_Term"
    Married = "Married"
    Loan_ID = "Loan_ID"
    Education = "Education"
    ApplicantIncome = "ApplicantIncome"
    CoapplicantIncome = "CoapplicantIncome"
    Property_Area = "Property_Area"
    # TODO: documentation of variables

    # TODO remove the .value
    @classmethod
    def num_features(cls):
        return [
            cls.Credit_History.value,
            cls.ApplicantIncome.value,
            cls.CoapplicantIncome.value,
            cls.LoanAmount.value,
            cls.Loan_Amount_Term.value
        ]

    @classmethod
    def cat_features(cls):
        return [
            cls.Dependents.value,
            cls.Gender.value,
            cls.Married.value,
            cls.Education.value,
            cls.Self_Employed.value,
            cls.Education.value
        ]

    @classmethod
    def target(cls):
        return cls.Loan_Status.value
