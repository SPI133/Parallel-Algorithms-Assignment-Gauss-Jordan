#include "mpi.h"
#include <cmath>
#include <iostream>
#include <random>

#define CONTINUOUS_COLUMNS 0
#define SHUFFLE 1

/*
Creates an nX(n+1) matrix corresponding to A 
and b concatenated (from Ax=b))
*/
double** matrix_2D(int n)
{
	double** matrix = new double* [n];
	for (int i = 0; i < n; i++)
	{
		matrix[i] = new double[n + 1];
	}

	std::uniform_real_distribution<double> unif(0, 100);
	std::default_random_engine re;

	for (int i = 0; i < n; i++)
		for (int j = 0; j <= n; j++)
			matrix[i][j] = unif(re);

	return matrix;
}

//Method to verify final array
void print_matrix(int n, int m, double** A, int rank)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			std::cout << rank << " " << A[i][j] << std::endl;
}

void swap(double** A, int row1, int col1, int row2, int col2) 
{
	double temp = A[row1][col1];
	A[row1][col1] = A[row2][col2];
	A[row2][col2] = temp;
}

double* get_column(int size, double** A, int column, double* col)
{
	for (int i = 0; i < size; i++)
		col[i] = A[i][column];
	return col;
}

void update_column(int size, double** A, int column, double* col)
{
	for (int i = 0; i < size; i++)
		A[i][column] = col[i];
}

void T_kk(int k, int n, double** A, int* piv)
{
	int col = n - k;
	double* a = new double[col];

	for (int i = 0; i < col; i++)
	{
		a[i] = A[i+k][k];
	}

	double current_max = abs(a[0]);
	int current_max_pos = k;
	for (int i = 1; i < col; ++i)
	{
		double absolute_value = abs(a[i]);
		if (current_max < absolute_value) 
		{
			current_max = absolute_value;
			current_max_pos = i + k;
		}
	}

	*piv = current_max_pos;
	swap(A, current_max_pos, k, k, k);
	
	delete[] a;
	a = NULL;
}

void T_kj(int k, int j, int n, double** A, int current_max_pos) {
	swap(A, current_max_pos, j, k, j);
	A[k][j] = A[k][j] / A[k][k];

	for (int i = 0; i < n; i++)
	{
		if (i != k) {
			A[i][j] = A[i][j] - A[i][k] * A[k][j];
		}
	}
}

int get_proc_of_column(int method, int k, int columns, int procs)
{
	switch (method)
	{
	case CONTINUOUS_COLUMNS:
		return k / columns;
	case SHUFFLE:
		return k % procs;
	}
}

void GaussJordan_kji(int n, double** A, int rank, int p, MPI_Comm comm, int method)
{
	int columns = n / p;
	int piv;
	double* col = new double[n];
	int proc_of_column;

	for (int k = 0; k < n; k++) 
	{
		proc_of_column = get_proc_of_column(method, k, columns, p);

		if (proc_of_column == rank)
		{
			T_kk(k, n, A, &piv);
			get_column(n, A, k, col);
		}
		
		MPI_Bcast(col, n, MPI_DOUBLE, proc_of_column, comm);
		MPI_Bcast(&piv, 1, MPI_DOUBLE, proc_of_column, comm);
		
		if (rank != proc_of_column)
		{
			update_column(n, A, k, col);
		}

		for (int j = k + 1; j < n; j++)
		{
			if (j / columns == rank)
			{
				T_kj(k, j, n, A, piv);
			}
		}

		if (rank == p - 1)
			T_kj(k, n, n, A, piv);
		
		//MPI_Barrier(MPI_COMM_WORLD);
	}
	
	
	delete[] col;
	col = NULL;
}

void solve(int n, double** A, int rank, int numprocs, int method, std::string method_name)
{
	MPI_Barrier(MPI_COMM_WORLD);
	double start = MPI_Wtime();

	GaussJordan_kji(n, A, rank, numprocs, MPI_COMM_WORLD, method);



	if (rank == numprocs - 1)
	{
		/*print only the solution for the 5 first and 5 last variables*/
		int sol_start = 5;
		int sol_end = n - 5;
		if (sol_end < sol_start)
		{
			sol_end = sol_start;
		}

		for (int i = 0; i < sol_start; i++)
		{
			std::cout << "x[" << i << "] = " << A[i][n] << std::endl;
		}
		
		for (int i = sol_end; i < n; i++)
		{
			std::cout << "x[" << i << "] = " << A[i][n] << std::endl;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	double end = MPI_Wtime();

	if (rank == 0)
	{
		std::cout << "Runtime of " << method_name << " for " << 
			numprocs << " processors = " << end - start << 
			std::endl << std::endl;
	}
}

/*Arguments are n and method. n is the amount of variables and method is 
0(continuous columns or 1(Shufffle).*/
int main(int argc, char* argv[]) 
{
	int n = atoi(argv[1]);
	int method = atoi(argv[2]);
	
	std::string method_name;
	
	switch(method)
	{
		case 0:
			method_name = "Continuous Columns";
			break;
		case 1:
			method_name = "Shuffle";
			break;
		default:
			std::cout << "Arguments are n and method. n is the" << 
			"amount of variables and method is 0(continuous columns) " <<
			"or 1(Shufffle).";
			return 0;
	}
	
	double** A = matrix_2D(n);

	int rank;
	int numprocs;

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	solve(n, A, rank, numprocs, method, method_name);
	
	MPI_Finalize();	

	for (int i = 0; i < n; i++)
	{
		delete[] A[i];
	}
	delete[] A;
	A = NULL;
	
	return 0;
}