
def write_module(module, m_name, writer, step):
    for p_name, values in module.named_parameters():
        writer.add_histogram(f'{m_name}.{p_name}', values, step)