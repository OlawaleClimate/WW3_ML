module W3SNLMLMD

  !use, intrinsic :: iso_fortran_env, only : sp => real32
  use W3GDATMD, only : SIG, NK, NTH, NSPEC, NSEAL, NSEA  ! Import parameters from W3GDATMD, including NSEAL and NSEA
  use W3PARALL, only : INIT_GET_ISEA  ! Import INIT_GET_ISEA subroutine
  USE CONSTANTS, ONLY: TPIINV
  USE ftorch
  
  implicit none

  private
  public :: ml_init, ml_routine, ml_final

  contains

  subroutine ml_init(mean_data, std_data, out_mean_data, out_std_data, rankk,  model)
    ! Load model and normalization data, and output them to be used in ml_routine

    real, dimension(1802), intent(out) :: mean_data, std_data
    real, dimension(1800), intent(out) :: out_mean_data, out_std_data
    integer, intent(in):: rankk
    type(torch_module), intent(out) :: model


    ! Load ML model
    model =torch_module_load('/pscratch/sd/o/olawale/Postdoc_NCAR/ML_models/ML_model_ts.pt')

    ! Load normalization data
    call load_data('/pscratch/sd/o/olawale/Postdoc_NCAR/in_norm_data_mean_nl2.txt', mean_data)
    call load_data('/pscratch/sd/o/olawale/Postdoc_NCAR/in_norm_data_std_nl2.txt', std_data)
    call load_data('/pscratch/sd/o/olawale/Postdoc_NCAR/out_norm_data_mean_nl2.txt', out_mean_data)
    call load_data('/pscratch/sd/o/olawale/Postdoc_NCAR/out_norm_data_std_nl2.txt', out_std_data)

  end subroutine ml_init

  subroutine ml_routine(DW, CG, VA, VNL_ML, mean_data, std_data, out_mean_data, out_std_data, rankk, model)
    ! Perform inference and normalization using model and normalization parameters

    real, intent(in) :: DW(NSEA), CG(1:NK, NSEA)
    real, intent(in) :: VA(1800, NSEAL)
    real, intent(out) :: VNL_ML(NK * NTH, NSEAL)
    real, dimension(1802), intent(in) :: mean_data, std_data
    real, dimension(1800), intent(in) :: out_mean_data, out_std_data
    integer, intent(in):: rankk
    type(torch_module), intent(in) :: model

    ! Local input and output arrays
    real :: input_array2(NSEAL, NSPEC + NK + 1)
    real :: output_array(NSEAL, NSPEC)
    real :: VA_C(NSPEC, NSEAL)  ! Local variable for converted data
    real :: FACTR(1800, NSEAL)  ! Local FACTR array

    ! Local Torch tensor structures
    type(torch_tensor), dimension(1) :: in_tensors
    type(torch_tensor) :: out_tensor

    integer :: JSEA, ISEA, IFR, ITH, ISP
    real :: CONX
        integer, parameter :: in_dims = 2, out_dims =2
    integer :: in_layout(in_dims), out_layout(out_dims)


    in_layout =  (/1,2/)
    out_layout = (/1,2/)

   VNL_ML = 0
   VA_C = 0

    !Normalize input data and fill input_array2
    do JSEA = 1, NSEAL
      call INIT_GET_ISEA(ISEA, JSEA)
      do IFR = 1, NK
        CONX = TPIINV / SIG(IFR) * CG(IFR, ISEA)
        do ITH = 1, NTH
          ISP = ITH + (IFR - 1) * NTH
          FACTR(ISP, JSEA) = CONX
        end do
      end do
      call row_to_column_major((VA(1:1800, JSEA) / FACTR(:,JSEA)), 50, 36, VA_C(1:1800, JSEA))
      input_array2(JSEA, NSPEC + 2:NSPEC + NK + 1) = (CG(1:NK, ISEA) - mean_data(1802)) / std_data(1802)
      input_array2(JSEA, NSPEC + 1) = (DW(ISEA) - mean_data(1801)) / std_data(1801)
      input_array2(JSEA, 1:NSPEC) = (VA_C(:, JSEA) - mean_data(1:1800)) / std_data(1:1800)
    end do

    ! Create Torch input and output tensors
    in_tensors(1) = torch_tensor_from_array(input_array2, in_layout, torch_kCPU)
    out_tensor = torch_tensor_from_array(output_array, out_layout, torch_kCPU)

    ! Run forward inference
    call torch_module_forward(model, in_tensors, 1, out_tensor)

    ! Denormalize output and post-process results
    do JSEA = 1, NSEAL
      output_array(JSEA, :) = (output_array(JSEA, :) * out_std_data(1:1800)) + out_mean_data(1:1800)
      call column_to_row_major(output_array(JSEA, :), 50, 36, VNL_ML(:, JSEA))
      VNL_ML(:, JSEA) = FACTR(:, JSEA) * VNL_ML(:, JSEA)
    end do
    
    ! Cleanup tensors
    call torch_tensor_delete(in_tensors(1))
    call torch_tensor_delete(out_tensor)

  end subroutine ml_routine

  subroutine ml_final(model)
    ! Cleanup model

    type(torch_module), intent(inout) :: model

    ! Delete model to release resources
    call torch_module_delete(model)
  end subroutine ml_final

  subroutine load_data(file_path, data)
    character(len=*), intent(in) :: file_path
    real, dimension(:), intent(out) :: data
    integer :: io_status

    open(49, file=file_path, status='old', access='stream', form='formatted')
    read(49, *, iostat=io_status) data
    close(49)
    if (io_status /= 0) then
      print *, "Error reading data from ", file_path
    end if
  end subroutine load_data

   subroutine row_to_column_major(row_flattened, rows, cols, column_flattened)
    implicit none
    integer, intent(in) :: rows, cols
    real, dimension(:), intent(in) :: row_flattened
    real, dimension(:), intent(out) :: column_flattened
    integer :: i, j, index


    ! Perform the conversion from row-major to column-major order
    index = 1
    do j = 1, cols
      do i = 1, rows
        column_flattened(index) = row_flattened((i-1) * cols + j)
        index = index + 1 
      end do
    end do
  end subroutine row_to_column_major

  ! Subroutine to convert column-major flattened to row-major flattened
  subroutine column_to_row_major(column_flattened, rows, cols, row_flattened)
    implicit none
    integer, intent(in) :: rows, cols
    real, dimension(:), intent(in) :: column_flattened
    real, dimension(:),  intent(out) :: row_flattened
    integer :: i, j, index

    ! Perform the conversion from column-major to row-major order
    index = 1
    do i = 1, rows
      do j = 1, cols
        row_flattened(index) = column_flattened((j-1) * rows + i)
        index = index + 1
      end do
    end do
  end subroutine column_to_row_major


end module W3SNLMLMD

