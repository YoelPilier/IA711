{
  description = "Flake para un entorno Jupyter + IA + Fish 💻🐟";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    unstable.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    unstable,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {inherit system;};
      unpkgs = import unstable {inherit system;};
    in {
      devShells.default = pkgs.mkShell {
        name = "impurejupyterenv";

        buildInputs = [
          (pkgs.python3.withPackages (ps:
            with ps; [
              jupyter
              ipykernel
              notebook
              pillow
              matplotlib
              torch
              numpy
              tqdm
              datasets
              safetensors
              torchmetrics
              torchvision
              lpips
            ]))
          pkgs.fish
        ];

        shellHook = ''
          exec fish
        '';
      };
    });
}
