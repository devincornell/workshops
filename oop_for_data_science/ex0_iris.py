from __future__ import annotations
import pandas as pd
import typing
import statistics
import sklearn.datasets

class Iris:
    def __init__(self, sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
        self.check_val(sepal_length)
        self.check_val(sepal_width)
        self.check_val(petal_length)
        self.check_val(petal_width)

        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width

    def __repr__(self):
        return (f'{self.__class__.__name__}(sepal_length={self.sepal_length}, '
            f'sepal_width={self.sepal_width}, petal_length={self.petal_length}, '
            f'petal_width={self.petal_width})')

    def check_val(self, val: float):
        '''Make sure val is not negative, raise ValueError if not.'''
        if val <= 0:
            raise ValueError('Dimension values cannot be equal to or less than zero.')

    def petal_area(self) -> float:
        '''Square-approximated surface area of petal.'''
        return self.petal_length * self.petal_width


class Irises(typing.List[Iris]):
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Irises:
        new: Irises = cls()
        for rowid, row in df.iterrows():
            new.append(Iris(
                sepal_length = row['sepal length (cm)'],
                sepal_width = row['sepal width (cm)'],
                petal_length = row['petal length (cm)'],
                petal_width = row['petal width (cm)'],
            ))
        return new

    def av_petal_length(self) -> float:
        '''Get the average sepal width.'''
        return statistics.mean(iris.petal_length for iris in self)

    def av_petal_area(self) -> float:
            '''Get the average sepal width.'''
            return statistics.mean(iris.petal_area() for iris in self)

if __name__ == '__main__':
    
    iris_df, target = sklearn.datasets.load_iris(return_X_y=True, as_frame=True)
    irises = Irises.from_dataframe(iris_df)
    
    # print out each iris
    for iris in irises:
        print(iris)

    print(f'{irises.av_petal_length()=}')
    print(f'{irises.av_petal_area()=}')

    


