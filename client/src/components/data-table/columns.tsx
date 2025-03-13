import React from 'react';
import { useTable, useSortBy } from 'react-table';


const data = [
  { language: 'English', explanation: 'The user expressed happiness.' },
  { language: 'Tagalog', explanation: 'Ang user ay nagpahayag ng kaligayahan.' },
  { language: 'English', explanation: 'The user seems frustrated due to the technical issue.' },
  { language: 'Tagalog', explanation: 'Mukhang frustrated ang user dahil sa teknikal na problema.' },
];

const columns = [
  {
    accessorKey: "language",
    header: "Language",
    cell: ({ row }) => <div>{row.getValue("language") || "Unknown"}</div>,
  },
  {
    accessorKey: "explanation",
    header: "Explanation",
    cell: ({ row }) => (
      <div className="max-w-[300px] truncate" title={row.getValue("explanation")}>
        {row.getValue("explanation") || "No explanation provided"}
      </div>
    ),
  },
];


function MyTable({ columns, data }) {
  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    rows,
    prepareRow,
  } = useTable({ columns, data }, useSortBy);

  return (
    <table {...getTableProps()}>
      <thead>
        {headerGroups.map(headerGroup => (
          <tr {...headerGroup.getHeaderGroupProps()}>
            {headerGroup.headers.map(column => (
              <th {...column.getHeaderProps(column.getSortByToggleProps())}>
                {column.render('header')}
                <span>
                  {column.isSorted
                    ? column.isSortedDesc
                      ? ' 🔽'
                      : ' 🔼'
                    : ''}
                </span>
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody {...getTableBodyProps()}>
        {rows.map((row, i) => {
          prepareRow(row);
          return (
            <tr {...row.getRowProps()}>
              {row.cells.map(cell => {
                return <td {...cell.getCellProps()}>{cell.render('cell')}</td>;
              })}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function App() {
  return (
    <div>
      <MyTable columns={columns} data={data} />
    </div>
  );
}

export default App;