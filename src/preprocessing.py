REMOVED_COLUMNS = [0, 2, 4, 5, 6]
COUNTRIES = ['Romania', 'Republic of Moldova', 'Ukraine', 'Serbia', 'Bulgaria', 'Hungary']
MONTHS = ['January', 'February', 'March', 'April', "May", "June", "July", "August", "September", "October", "November",
          "December"]


def processData():
    processedRecords = []
    header = []
    with open('../Environment_Temperature_change_E_All_Data_NOFLAG.csv', 'r') as file:
        header = file.readline().split(",")
        header = removeColumns(header, REMOVED_COLUMNS)

        while True:
            line = file.readline()
            if not line:
                break

            line = line.replace('\"', '')
            record = line.split(',')

            if passesFilters(record):
                processedRecords.append(removeColumns(record, REMOVED_COLUMNS))

    with open('output.csv', 'w') as file:
        file.write(recordToCSV(header))
        for record in processedRecords:
            file.write(recordToCSV(record))


def passesFilters(record):
    return filterByCountry(record) and filterByMonth(record) and filterByElement(record)


# Filter for countries that are not neighbours of Romania
def filterByCountry(record):
    return record[1] in COUNTRIES

# Filter for rows that have more than one month in this field
def filterByMonth(record):
    return record[3] in MONTHS


# Filter for rows that don't contain 'Temperature change' values
def filterByElement(record):
    return record[5] == "Temperature change"


# Remove irrelevant columns
#   - 'code' columns
#   - 'Element' column, because it is always 'Temperature change'
def removeColumns(record, columns):
    for column in sorted(columns, reverse=True):
        del record[column]
    return record


def recordToCSV(record):
    return ','.join(record)


if __name__ == '__main__':
    processData()
