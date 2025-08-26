FROM julia:1.10.2-bookworm

WORKDIR /app
COPY Project.toml Manifest.toml ./

RUN julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

COPY . .

EXPOSE 8000
CMD ["julia", "src/main.jl"]