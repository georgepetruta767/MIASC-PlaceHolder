REMOVED_COLUMNS = [0, 1, 2, 3, 8]


def processData():
    processedRecords = []
    header = None
    for year in range(1961, 2016):
        with open(f'../temps/climrbsn{year}.csv', 'r') as file:
            if header is None:
                header = file.readline().split(",")
                header = remove_columns(header, REMOVED_COLUMNS)
            else:
                file.readline()

            while True:
                line = file.readline()
                if not line:
                    break

                record = line.split(',')
                record = [entry.strip() for entry in record]

                if passes_filters(record):
                    processedRecords.append(remove_columns(record, REMOVED_COLUMNS))

    with open('output.csv', 'w') as file:
        file.write(record_to_CSV(header))
        for record in processedRecords:
            file.write(record_to_CSV(record))


def passes_filters(record):
    return filter_by_station(record)


# Filter for a single meteo station
def filter_by_station(record):
    return record[0] == '15346'


# Remove irrelevant columns
def remove_columns(record, columns):
    for column in sorted(columns, reverse=True):
        del record[column]
    return record


def record_to_CSV(record):
    return ','.join(record) + '\n'


if __name__ == '__main__':
    processData()
