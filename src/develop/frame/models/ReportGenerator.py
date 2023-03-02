import datetime
from tabulate import tabulate
import pandas as pd

class ReportGenerator(object):

    def __init__(self, outfile):
        self.outfile = outfile


    def add_header(self, description):
        print(f"""
=============================================================================================================================================================================== 
                                                                     REPORT: MODELS PERFORMANCE
=============================================================================================================================================================================== 
Last Updated On: {
    datetime.date.strftime(datetime.date.today(), "%Y-%m-%d")
}

Description: {description}
"""
        ,file=self.outfile)


    def add_line(self, text):
        print(f"{text}", file=self.outfile)

    def insert_break(self):
        print(
        f"___________________________________________________________________________________________________________________________________________________________________________\n"
            ,file=self.outfile
        )

    def tabule_df(self, data, format, showindex=False):
        print(
            tabulate(
                data,
                headers='keys',
                showindex=showindex,
                tablefmt=format
            ),
            file=self.outfile
        )
