import nbformat as nbf

# Function to create a notebook from an .m file
def create_notebook_from_m_file(m_file_path, notebook_path):
    # Read the .m file
    with open(m_file_path, 'r') as f:
        m_code = f.read()

    # Create a new Jupyter notebook
    nb = nbf.v4.new_notebook()

    # Split the code into individual cells (if necessary)
    code_cells = m_code.split('\n\n')  # Assume code blocks are separated by empty lines

    # Add each block of code as a separate code cell
    for code_cell in code_cells:
        nb.cells.append(nbf.v4.new_code_cell(code_cell))

    # Write the notebook to a file
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)

# Example usage
create_notebook_from_m_file('ex5.m', 'first_notebook_2025.ipynb')
