
def processData():
    with open('Environment_Temperature_change_E_All_Data_NOFLAG.csv', 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break

            records = line.split(',')
            if filterByCountry(records):


def filterByCountry(records):
    relatedCountries = ['Romania', 'Republic of Moldova', 'Ukraine', 'Serbia', 'Bulgaria', 'Hungary']

    return records[1] in relatedCountries



